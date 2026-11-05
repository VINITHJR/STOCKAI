"""Train StockAI's shared multi-stock LSTM v2 model.

Run from the StockAI project root:
python -m backend.app.ml.training_script --instrument-keys "NSE_EQ|KEY1" "NSE_EQ|KEY2" ... --days 1095

By default, this script writes a candidate under models/candidates and does not
replace the artifacts used by the API. Add --write-production only after independent
walk-forward validation. Legacy H5 artifacts are never touched.
"""
from __future__ import annotations

import argparse
from datetime import date, timedelta
import json
from pathlib import Path
from typing import Dict, List

from backend.app.services.upstox_service import get_historical_candles
from backend.app.ml.lstm_model import LSTMPricePredictor, TENSORFLOW_AVAILABLE


def fetch_training_data(instrument_key: str, days: int = 365) -> List[dict]:
    to_date = date.today()
    from_date = to_date - timedelta(days=int(days))
    result = get_historical_candles(
        instrument_key=instrument_key, unit="days", interval="1",
        to_date=to_date.isoformat(), from_date=from_date.isoformat(),
    )
    raw = result.get("data", {}).get("candles", []) if isinstance(result, dict) else []
    candles = []
    for row in raw:
        if not isinstance(row, (list, tuple)) or len(row) < 6:
            continue
        try:
            candles.append({"timestamp": row[0], "open": float(row[1]), "high": float(row[2]),
                            "low": float(row[3]), "close": float(row[4]), "volume": float(row[5] or 0)})
        except (TypeError, ValueError):
            continue
    if not candles:
        raise ValueError(f"No valid historical candles returned for {instrument_key}.")
    return candles


def train_model(
    instrument_keys: List[str], days: int = 365, epochs: int = 40,
    batch_size: int = 32, write_production: bool = False,
) -> Dict:
    if not TENSORFLOW_AVAILABLE:
        raise RuntimeError("TensorFlow is not available. Install a compatible TensorFlow build before training.")
    if int(days) < 180:
        raise ValueError("Use at least 180 calendar days of history; 730-1095 days is preferable for evaluation.")
    if int(epochs) < 1 or int(batch_size) < 1:
        raise ValueError("epochs and batch_size must be positive integers.")
    keys = list(dict.fromkeys(k.strip() for k in instrument_keys if k and k.strip()))
    if len(keys) < 2:
        raise ValueError("Pass at least 2 different --instrument-keys. For a general shared model, use 8-20 instruments where possible.")
    datasets = []
    errors = {}
    for index, key in enumerate(keys, start=1):
        print(f"[{index}/{len(keys)}] Fetching {days} calendar days for {key} ...")
        try:
            candles = fetch_training_data(key, days)
            datasets.append({"instrument_key": key, "candles": candles})
            print(f"  Loaded {len(candles)} candles")
        except Exception as exc:
            errors[key] = str(exc)
            print(f"  SKIP: {exc}")
    if len(datasets) < 2:
        raise RuntimeError(f"At least two instruments must fetch successfully. Fetch errors: {errors}")
    # Candidate-only is the safe default: don't replace the model used by the API
    # until the candidate has passed an independent walk-forward evaluation.
    model_path = (
        "backend/app/ml/models"
        if write_production
        else "backend/app/ml/models/candidates/lstm_v2_candidate"
    )
    predictor = LSTMPricePredictor(
        sequence_length=60, prediction_horizons=[1, 3, 7], model_path=model_path
    )
    metrics = predictor.train_multi_stock(datasets, epochs=epochs, batch_size=batch_size)
    metrics["artifact_mode"] = "production" if write_production else "candidate_only"
    metrics["production_artifacts_modified"] = bool(write_production)
    metrics["requested_instruments"] = keys
    metrics["fetch_errors"] = errors
    report_path = Path("backend/app/ml/models/lstm_v2_training_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    print("\nTraining completed.")
    print(f"Model:   {predictor.model_file}")
    print(f"Scalers: {predictor.scaler_file}")
    print(f"Report:  {report_path}")
    print(json.dumps({k: v for k, v in metrics.items() if k not in {"loss", "val_loss", "mae", "val_mae"}}, indent=2, default=str))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train StockAI multi-stock LSTM v2")
    parser.add_argument("--instrument-keys", nargs="+", required=True, help="At least 2 Upstox instrument keys, quoted individually")
    parser.add_argument("--days", type=int, default=365, help="Calendar days of history per instrument (default: 365)")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--write-production", action="store_true",
        help="Explicitly write to the live v2 artifact paths. Only use after independent validation.",
    )
    args = parser.parse_args()
    train_model(
        args.instrument_keys, days=args.days, epochs=args.epochs,
        batch_size=args.batch_size, write_production=args.write_production,
    )


if __name__ == "__main__":
    main()
