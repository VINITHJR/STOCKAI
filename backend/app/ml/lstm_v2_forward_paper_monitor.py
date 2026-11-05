"""Read-only StockAI LSTM v2 forecast monitor.

Loads saved model/scaler pairs without training or modifying them. On each run,
it records a forecast for the latest available post-cutoff candle per instrument
and fills in realized returns as future daily candles become available.

Only writes: backend/app/ml/models/research/lstm_v2_forward_paper_log.csv
"""

from __future__ import annotations

import hashlib
import os
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backend.app.ml.lstm_model import LSTMPricePredictor
from backend.app.ml.training_script import fetch_training_data

ROOT = Path("backend/app/ml/models")
KEY_FILE = ROOT / "lstm_instrument_keys.txt"
LOG_FILE = ROOT / "research" / "lstm_v2_forward_paper_log.csv"

# Conservative provisional cutoff based on the candidate training report and
# artifact timestamps. Update only if stronger training-date evidence appears.
CUTOFF_DATE = date(2026, 10, 10)
HORIZONS = (1, 3, 7)  # trading candles, not calendar days
FETCH_DAYS = 365
RAW_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]

MODEL_DIRS = {
    "ACTIVE_PRODUCTION": ROOT,
    "CANDIDATE_WITH_ITS_SCALER": ROOT / "candidates" / "lstm_v2_repaired_candidate",
}

BASE_COLUMNS = [
    "pair_id",
    "model_label",
    "model_sha256",
    "scaler_sha256",
    "instrument_key",
    "base_date",
    "base_close",
    "prediction_recorded_at_utc",
]
HORIZON_COLUMNS = []
for horizon in HORIZONS:
    HORIZON_COLUMNS.extend(
        [f"predicted_return_{horizon}d", f"target_date_{horizon}d",
         f"actual_close_{horizon}d", f"actual_return_{horizon}d"]
    )
COLUMNS = BASE_COLUMNS + HORIZON_COLUMNS


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def local_dates(df: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(df["timestamp"], utc=True, errors="coerce").dt.tz_convert(
        "Asia/Kolkata"
    )


def load_instrument_keys() -> list[str]:
    if not KEY_FILE.is_file():
        raise FileNotFoundError(f"Instrument file not found: {KEY_FILE}")
    keys = []
    for line in KEY_FILE.read_text(encoding="utf-8").splitlines():
        key = line.split("#", 1)[0].strip()
        if key and key not in keys:
            keys.append(key)
    if not keys:
        raise RuntimeError(f"No instrument keys found in {KEY_FILE}")
    return keys


def load_predictors() -> dict[str, dict[str, Any]]:
    predictors: dict[str, dict[str, Any]] = {}
    for label, model_dir in MODEL_DIRS.items():
        model_file = model_dir / "lstm_model_v2.keras"
        scaler_file = model_dir / "lstm_scalers_v2.pkl"
        if not model_file.is_file() or not scaler_file.is_file():
            raise FileNotFoundError(
                f"Missing model/scaler for {label}: {model_file} / {scaler_file}"
            )

        predictor = LSTMPricePredictor(
            sequence_length=60,
            prediction_horizons=list(HORIZONS),
            model_path=str(model_dir),
        )
        predictor.load()
        if predictor.model is None:
            raise RuntimeError(f"Model failed to load: {label}")
        if not hasattr(predictor.feature_scaler, "n_features_in_"):
            raise RuntimeError(f"Scaler failed to load: {label}")

        model_hash = sha256_file(model_file)
        scaler_hash = sha256_file(scaler_file)
        pair_id = hashlib.sha256(f"{model_hash}:{scaler_hash}".encode()).hexdigest()
        predictors[label] = {
            "predictor": predictor,
            "model_sha256": model_hash,
            "scaler_sha256": scaler_hash,
            "pair_id": pair_id,
        }
        print(
            f"Loaded {label}: features={predictor.feature_scaler.n_features_in_}, "
            f"scaler_samples={predictor.feature_scaler.n_samples_seen_}, "
            f"pair_id={pair_id[:12]}"
        )
    return predictors


def load_log() -> pd.DataFrame:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not LOG_FILE.exists():
        return pd.DataFrame(columns=COLUMNS)
    log = pd.read_csv(LOG_FILE)
    for column in COLUMNS:
        if column not in log.columns:
            log[column] = np.nan
    return log[COLUMNS].copy()


def make_prediction(
    predictor: LSTMPricePredictor,
    clean_df: pd.DataFrame,
) -> np.ndarray:
    # Rebuild features from history available at this base candle only.
    prefix = clean_df[RAW_COLUMNS].to_dict(orient="records")
    prefix_df = predictor._clean_candles(prefix)
    raw_features = np.asarray(predictor._relative_features(prefix_df), dtype=np.float64)

    if raw_features.ndim != 2 or raw_features.shape[0] < predictor.sequence_length:
        raise ValueError("Insufficient rows after feature engineering")
    if raw_features.shape[1] != predictor.feature_scaler.n_features_in_:
        raise ValueError(f"Unexpected feature shape: {raw_features.shape}")

    raw_sequence = raw_features[-predictor.sequence_length:]
    if not np.isfinite(raw_sequence).all():
        raise ValueError("Non-finite input features")

    scaled = predictor.feature_scaler.transform(raw_sequence)
    prediction = np.asarray(
        predictor.model.predict(scaled[None, ...], verbose=0), dtype=np.float64
    ).reshape(-1)
    if len(prediction) != len(HORIZONS) or not np.isfinite(prediction).all():
        raise ValueError(f"Invalid model output: {prediction}")
    # Match the clipping used by the production prediction method.
    return np.clip(prediction, -0.5, 0.5)


def main() -> None:
    print("=" * 76)
    print("STOCKAI LSTM V2 FORWARD PAPER MONITOR")
    print("=" * 76)
    print(f"Provisional cutoff: {CUTOFF_DATE} (Asia/Kolkata)")
    print("No model training or production-artifact writes will occur.")
    print(f"Research log: {LOG_FILE}")

    keys = load_instrument_keys()
    predictors = load_predictors()
    log = load_log()
    if not log.empty:
        for column in ("pair_id", "instrument_key", "base_date"):
            log[column] = log[column].astype(str)

    existing = set()
    if not log.empty:
        existing = set(
            zip(log["pair_id"].astype(str), log["instrument_key"].astype(str), log["base_date"].astype(str))
        )

    market_data: dict[str, dict[str, Any]] = {}
    new_rows: list[dict[str, Any]] = []
    failures: list[tuple[str, str]] = []

    print(f"Instruments: {len(keys)} | latest-candle forecast per instrument/pair")

    for index, key in enumerate(keys, start=1):
        print(f"\n[{index}/{len(keys)}] {key}")
        try:
            candles = fetch_training_data(key, days=FETCH_DAYS)
            clean_df = predictors["ACTIVE_PRODUCTION"]["predictor"]._clean_candles(candles)
            clean_df = clean_df.reset_index(drop=True)
            if len(clean_df) < 60 + max(HORIZONS):
                raise ValueError(f"Insufficient clean candles: {len(clean_df)}")

            dates = local_dates(clean_df)
            closes = pd.to_numeric(clean_df["close"], errors="coerce").to_numpy(dtype=float)
            if dates.isna().any() or not np.isfinite(closes).all():
                raise ValueError("Invalid date or close values in cleaned candle data")

            date_strings = [stamp.date().isoformat() for stamp in dates]
            if len(set(date_strings)) != len(date_strings):
                raise ValueError("Duplicate India-local candle dates after cleaning")

            market_data[key] = {
                "df": clean_df,
                "dates": dates,
                "date_to_index": {day: i for i, day in enumerate(date_strings)},
                "closes": closes,
                "date_strings": date_strings,
            }

            latest_index = len(clean_df) - 1
            latest_date = date_strings[latest_index]
            latest_close = float(closes[latest_index])
            print(
                f"  clean candles={len(clean_df)}, "
                f"range={date_strings[0]}..{latest_date}"
            )

            if date.fromisoformat(latest_date) <= CUTOFF_DATE:
                print("  latest candle is not after cutoff; skipped")
                continue

            for label, info in predictors.items():
                pair_id = info["pair_id"]
                unique_key = (pair_id, key, latest_date)
                if unique_key in existing:
                    print(f"  {label}: latest date already recorded")
                    continue

                prediction = make_prediction(info["predictor"], clean_df.iloc[:latest_index + 1])
                row: dict[str, Any] = {
                    "pair_id": pair_id,
                    "model_label": label,
                    "model_sha256": info["model_sha256"],
                    "scaler_sha256": info["scaler_sha256"],
                    "instrument_key": key,
                    "base_date": latest_date,
                    "base_close": latest_close,
                    "prediction_recorded_at_utc": datetime.now(timezone.utc).isoformat(),
                }
                for j, horizon in enumerate(HORIZONS):
                    row[f"predicted_return_{horizon}d"] = float(prediction[j])
                    row[f"target_date_{horizon}d"] = ""
                    row[f"actual_close_{horizon}d"] = np.nan
                    row[f"actual_return_{horizon}d"] = np.nan
                new_rows.append(row)
                existing.add(unique_key)
                print(
                    f"  {label}: forecast recorded for {latest_date}; "
                    + ", ".join(
                        f"{h}d={100.0 * prediction[j]:+.3f}%"
                        for j, h in enumerate(HORIZONS)
                    )
                )

        except Exception as exc:
            failures.append((key, f"{type(exc).__name__}: {exc}"))
            print(f"  WARNING: {type(exc).__name__}: {exc}")
        time.sleep(0.15)

    if new_rows:
        log = pd.concat([log, pd.DataFrame(new_rows)], ignore_index=True)

    # Update realized returns for every recorded forecast whose target candles
    # are now available. Horizons count observed trading candles in the series.
    if not log.empty:
        for row_index, row in log.iterrows():
            key = str(row.get("instrument_key", ""))
            data = market_data.get(key)
            if data is None:
                continue

            base_date = str(row.get("base_date", ""))
            base_index = data["date_to_index"].get(base_date)
            if base_index is None:
                continue

            base_close = pd.to_numeric(row.get("base_close"), errors="coerce")
            if not np.isfinite(base_close) or base_close <= 0:
                base_close = data["closes"][base_index]

            for horizon in HORIZONS:
                target_index = base_index + horizon
                if target_index >= len(data["closes"]):
                    continue

                target_close = float(data["closes"][target_index])
                if not np.isfinite(target_close) or target_close <= 0:
                    continue

                log.at[row_index, f"target_date_{horizon}d"] = data["date_strings"][target_index]
                log.at[row_index, f"actual_close_{horizon}d"] = target_close
                log.at[row_index, f"actual_return_{horizon}d"] = target_close / float(base_close) - 1.0

        log = log.drop_duplicates(
            subset=["pair_id", "instrument_key", "base_date"], keep="last"
        )
        log = log[COLUMNS].sort_values(
            ["model_label", "instrument_key", "base_date"]
        ).reset_index(drop=True)

        temp_file = LOG_FILE.with_suffix(".csv.tmp")
        log.to_csv(temp_file, index=False)
        os.replace(temp_file, LOG_FILE)

    print("\n" + "=" * 76)
    print("MATURED PAPER-EVALUATION RESULTS")
    print("=" * 76)
    if log.empty:
        print("No forecast records exist yet.")
    else:
        for label in MODEL_DIRS:
            pair_rows = log[log["model_label"] == label]
            print(f"\n{label}: total logged forecasts={len(pair_rows)}")
            for horizon in HORIZONS:
                actual_col = f"actual_return_{horizon}d"
                pred_col = f"predicted_return_{horizon}d"
                mature = pair_rows[
                    pd.to_numeric(pair_rows[actual_col], errors="coerce").notna()
                    & pd.to_numeric(pair_rows[pred_col], errors="coerce").notna()
                ]
                if mature.empty:
                    print(f"  {horizon}d: no matured outcomes yet")
                    continue

                pred = pd.to_numeric(mature[pred_col]).to_numpy(dtype=float) * 100.0
                actual = pd.to_numeric(mature[actual_col]).to_numpy(dtype=float) * 100.0
                model_mae = float(np.mean(np.abs(pred - actual)))
                baseline_mae = float(np.mean(np.abs(actual)))
                improvement = (
                    100.0 * (baseline_mae - model_mae) / baseline_mae
                    if baseline_mae > 0 else float("nan")
                )
                bias = float(np.mean(pred - actual))
                corr = (
                    float(np.corrcoef(pred, actual)[0, 1])
                    if np.std(pred) > 0 and np.std(actual) > 0 else float("nan")
                )
                sign_accuracy = float(100.0 * np.mean(np.sign(pred) == np.sign(actual)))
                print(
                    f"  {horizon}d: n={len(mature)}, "
                    f"dates={mature['base_date'].nunique()}, "
                    f"stocks={mature['instrument_key'].nunique()}, "
                    f"model_MAE={model_mae:.5f} pp, baseline_MAE={baseline_mae:.5f} pp, "
                    f"improvement={improvement:+.2f}%, bias={bias:+.5f} pp, "
                    f"correlation={corr:.5f}, sign_accuracy={sign_accuracy:.2f}%"
                )

    if failures:
        print("\nFETCH / EVALUATION WARNINGS:")
        for key, message in failures:
            print(f"  {key}: {message}")

    print(f"\nResearch log updated: {LOG_FILE}")
    print("This log is a research artifact only; production models and scalers were not changed.")
    print("Run this script after each completed daily candle is available to collect further outcomes.")


if __name__ == "__main__":
    main()
