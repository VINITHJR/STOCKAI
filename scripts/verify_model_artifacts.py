#!/usr/bin/env python3

"""
StockAI - Production Model Artifact Verification

Checks:
1. SHA-256 integrity of production model artifacts.
2. LSTM model loading and synthetic inference.
3. Random Forest model loading and synthetic inference.
4. Artifact integrity after inference.

This script does NOT:
- Train or retrain models.
- Save or overwrite model artifacts.
- Call live market APIs.
- Access final holdout data.
"""

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MODEL_DIR = PROJECT_ROOT / "backend" / "app" / "ml" / "models"


# ---------------------------------------------------------
# EXPECTED PRODUCTION ARTIFACT HASHES
# ---------------------------------------------------------

EXPECTED_HASHES = {
    "lstm_model_v2.keras":
        "8a89b3c179bec7c33aa049558e107e71fd30060fae6f774a324e39c0c98bacbf",

    "lstm_scalers_v2.pkl":
        "1a9a8fa47c97c76d26fae97ec022f7c3cb3d6c523f1f63444617a5e6f614ebf1",

    "rf_trend_model.pkl":
        "2dc9a7925925144c8ad66df23843a1c28330667bd4f5bf72390073e3ea3f98a9",
}


# ---------------------------------------------------------
# UTILITY FUNCTIONS
# ---------------------------------------------------------

def calculate_sha256(file_path: Path) -> str:
    """Calculate SHA-256 without loading the entire file into memory."""

    sha256 = hashlib.sha256()

    with file_path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def verify_artifact_hashes() -> None:
    """Verify all production model artifacts against known baselines."""

    print("\n===== VERIFY MODEL ARTIFACT HASHES =====")

    for filename, expected_hash in EXPECTED_HASHES.items():
        artifact_path = MODEL_DIR / filename

        if not artifact_path.is_file():
            raise FileNotFoundError(
                f"Missing model artifact: {artifact_path}"
            )

        actual_hash = calculate_sha256(artifact_path)

        if actual_hash != expected_hash:
            raise RuntimeError(
                f"SHA-256 mismatch for {filename}\n"
                f"Expected: {expected_hash}\n"
                f"Actual:   {actual_hash}"
            )

        print(f"PASS: {filename}")

    print("All production artifact hashes verified.")


def generate_synthetic_data():
    """
    Generate deterministic synthetic OHLCV data.

    This is test data, not actual market data.
    """

    rng = np.random.default_rng(42)

    row_count = 180
    steps = np.arange(row_count)

    returns = (
        0.0003
        + 0.008 * np.sin(2 * np.pi * steps / 25)
        + rng.normal(0, 0.003, row_count)
    )

    close = 1000 * np.exp(np.cumsum(returns))

    open_price = np.concatenate(
        ([close[0] * 0.999], close[:-1])
    )

    spread = rng.uniform(0.001, 0.006, row_count)

    high = np.maximum(open_price, close) * (1 + spread)
    low = np.minimum(open_price, close) * (1 - spread)

    volume = rng.integers(
        100_000,
        1_000_000,
        row_count
    )

    dates = pd.date_range(
        "2025-01-01",
        periods=row_count,
        freq="B",
        tz="Asia/Kolkata"
    )

    candles = [
        {
            "timestamp": dates[i].isoformat(),
            "open": float(open_price[i]),
            "high": float(high[i]),
            "low": float(low[i]),
            "close": float(close[i]),
            "volume": int(volume[i]),
        }
        for i in range(row_count)
    ]

    frame = pd.DataFrame({
        "timestamp": dates,
        "open": open_price,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })

    return candles, frame, float(close[-1])


# ---------------------------------------------------------
# LSTM INFERENCE TEST
# ---------------------------------------------------------

def test_lstm_inference(candles, latest_close):
    print("\n===== LSTM MODEL INFERENCE =====")

    from backend.app.ml.lstm_model import LSTMPricePredictor

    model = LSTMPricePredictor()

    if model.load() is not True:
        raise RuntimeError("LSTM model failed to load.")

    result = model.predict(candles, latest_close)

    if result.get("status") != "success":
        raise RuntimeError(
            f"LSTM prediction failed: {result}"
        )

    predictions = result.get("predictions")

    if not isinstance(predictions, list) or not predictions:
        raise RuntimeError("LSTM returned no predictions.")

    horizons = [
        prediction.get("horizon_sessions")
        for prediction in predictions
    ]

    if horizons != [1, 3, 7]:
        raise RuntimeError(
            f"Unexpected LSTM prediction horizons: {horizons}"
        )

    for prediction in predictions:
        predicted_price = float(prediction["predicted_price"])
        lower_bound = float(prediction["lower_bound"])
        upper_bound = float(prediction["upper_bound"])
        change_percent = float(prediction["change_percent"])

        if not all(np.isfinite(value) for value in (
            predicted_price,
            lower_bound,
            upper_bound,
            change_percent,
        )):
            raise RuntimeError(
                "LSTM returned a non-finite prediction value."
            )

        if lower_bound > predicted_price or predicted_price > upper_bound:
            raise RuntimeError(
                "LSTM prediction is outside its reported bounds."
            )

        print(
            f"{prediction['horizon_sessions']} sessions | "
            f"Direction: {prediction['direction']} | "
            f"Change: {change_percent:.4f}%"
        )

    print("LSTM INFERENCE: PASS")


# ---------------------------------------------------------
# RANDOM FOREST INFERENCE TEST
# ---------------------------------------------------------

def test_random_forest_inference(frame):
    print("\n===== RANDOM FOREST INFERENCE =====")

    from backend.app.ml.rf_trend_classifier import RFTrendClassifier

    model = RFTrendClassifier()

    if model.load_model() is not True:
        raise RuntimeError("Random Forest model failed to load.")

    result = model.predict_trend(frame)

    if result.get("trend") not in {
        "bearish",
        "neutral",
        "bullish",
    }:
        raise RuntimeError(
            f"Unexpected Random Forest trend: {result.get('trend')}"
        )

    probabilities = result.get("probabilities")

    if not isinstance(probabilities, dict) or not probabilities:
        raise RuntimeError(
            "Random Forest returned invalid probabilities."
        )

    probability_values = [
        float(value) for value in probabilities.values()
    ]

    if not all(
        np.isfinite(value) and 0 <= value <= 1
        for value in probability_values
    ):
        raise RuntimeError(
            "Random Forest returned an invalid probability value."
        )

    if not np.isclose(sum(probability_values), 1.0, atol=1e-5):
        raise RuntimeError(
            f"Random Forest probabilities do not sum to 1: "
            f"{sum(probability_values)}"
        )

    confidence = float(result["confidence"])

    if not np.isfinite(confidence) or not 0 <= confidence <= 1:
        raise RuntimeError(
            f"Invalid Random Forest confidence: {confidence}"
        )

    print("Trend:", result["trend"])
    print("Confidence:", confidence)
    print("Probabilities:", probabilities)

    print("RANDOM FOREST INFERENCE: PASS")


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    print("=" * 65)
    print("STOCKAI - PRODUCTION MODEL ARTIFACT VERIFICATION")
    print("=" * 65)

    print("Project root:", PROJECT_ROOT)
    print("Model directory:", MODEL_DIR)

    # Verify files before inference.
    verify_artifact_hashes()

    # Generate synthetic test inputs.
    print("\n===== GENERATE SYNTHETIC OHLCV DATA =====")

    candles, frame, latest_close = generate_synthetic_data()

    print("Synthetic rows:", len(frame))
    print("Latest synthetic close:", latest_close)

    # Test both production models.
    test_lstm_inference(candles, latest_close)
    test_random_forest_inference(frame)

    # Verify files again after inference.
    print("\n===== VERIFY ARTIFACTS AFTER INFERENCE =====")

    verify_artifact_hashes()

    print("\n" + "=" * 65)
    print("FINAL RESULT: ALL CHECKS PASSED")
    print("Models loaded and synthetic inference completed.")
    print("Production model artifacts remain unchanged.")
    print("No training or live API calls were performed.")
    print("No final holdout data was used.")
    print("=" * 65)


if __name__ == "__main__":
    main()
