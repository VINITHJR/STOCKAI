"""
Training script for Random Forest Trend Classification Model

Usage:
    python -m backend.app.ml.train_rf_model --instrument_key "NSE_EQ|INE002A01018" --days 365
"""

import argparse
import sys
from datetime import datetime, timedelta
from backend.app.services.upstox_service import get_historical_candles
from backend.app.ml.rf_trend_classifier import RFTrendClassifier
import pandas as pd


def fetch_training_data(instrument_key: str, days: int = 365):
    """
    Fetch historical data for training.

    Args:
        instrument_key: Upstox instrument key
        days: Number of days of historical data

    Returns:
        DataFrame with OHLCV data
    """
    print(f"Fetching {days} days of historical data for {instrument_key}...")

    to_date = datetime.now().date()
    from_date = to_date - timedelta(days=days)

    result = get_historical_candles(
        instrument_key=instrument_key,
        unit="days",
        interval="1",
        to_date=to_date.isoformat(),
        from_date=from_date.isoformat()
    )

    raw_candles = result.get("data", {}).get("candles", [])

    if not raw_candles:
        raise ValueError("No historical data available")

    # Convert to DataFrame
    data = []
    for candle in raw_candles:
        if len(candle) < 6:
            continue

        data.append({
            'timestamp': candle[0],
            'open': float(candle[1]),
            'high': float(candle[2]),
            'low': float(candle[3]),
            'close': float(candle[4]),
            'volume': float(candle[5]) if candle[5] else 0
        })

    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    print(f"Fetched {len(df)} candles")
    return df


def train_model(
    instrument_key: str,
    days: int = 365,
    forward_days: int = 5,
    n_estimators: int = 100,
    max_depth: int = 10
):
    """
    Train Random Forest model for trend classification.

    Args:
        instrument_key: Upstox instrument key
        days: Days of historical data to use
        forward_days: Days ahead to predict trend
        n_estimators: Number of trees in forest
        max_depth: Maximum depth of trees
    """
    print("="*60)
    print("Random Forest Trend Classification Model Training")
    print("="*60)
    print(f"Instrument: {instrument_key}")
    print(f"Historical days: {days}")
    print(f"Forward prediction days: {forward_days}")
    print(f"N estimators: {n_estimators}")
    print(f"Max depth: {max_depth}")
    print("="*60)

    # Fetch data
    df = fetch_training_data(instrument_key, days)

    # Initialize model
    classifier = RFTrendClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth
    )

    # Train
    metrics = classifier.train(
        df=df,
        forward_days=forward_days,
        test_size=0.2,
        random_state=42
    )

    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Train accuracy: {metrics['train_accuracy']:.2%}")
    print(f"Test accuracy: {metrics['test_accuracy']:.2%}")
    print(f"Train samples: {metrics['train_samples']}")
    print(f"Test samples: {metrics['test_samples']}")
    print("\nClass distribution:")
    print(f"  Bearish: {metrics['class_distribution']['bearish']}")
    print(f"  Neutral: {metrics['class_distribution']['neutral']}")
    print(f"  Bullish: {metrics['class_distribution']['bullish']}")

    print("\nTop 5 important features:")
    importance = sorted(
        metrics['feature_importance'].items(),
        key=lambda x: x[1],
        reverse=True
    )[:5]
    for feat, imp in importance:
        print(f"  {feat}: {imp:.4f}")
    print("="*60)

    # Save model
    classifier.save_model()

    # Test prediction on recent data
    print("\nTesting prediction on recent data...")
    current_price = df.iloc[-1]['close']

    prediction = classifier.predict_trend(df)

    print(f"\nCurrent Price: ₹{current_price:.2f}")
    print(f"\nTrend Prediction:")
    print(f"  Direction: {prediction['trend'].upper()}")
    print(f"  Confidence: {prediction['confidence']:.2%}")
    print(f"\nProbabilities:")
    print(f"  Bearish: {prediction['probabilities']['bearish']:.2%}")
    print(f"  Neutral: {prediction['probabilities']['neutral']:.2%}")
    print(f"  Bullish: {prediction['probabilities']['bullish']:.2%}")

    print("\n✅ Model saved successfully!")
    print(f"Model location: backend/app/ml/models/rf_trend_model.pkl")


def main():
    parser = argparse.ArgumentParser(
        description="Train Random Forest Trend Classification Model"
    )

    parser.add_argument(
        "--instrument_key",
        type=str,
        default="NSE_EQ|INE002A01018",
        help="Upstox instrument key (default: Reliance)"
    )

    parser.add_argument(
        "--days",
        type=int,
        default=365,
        help="Days of historical data (default: 365)"
    )

    parser.add_argument(
        "--forward_days",
        type=int,
        default=5,
        help="Days ahead to predict trend (default: 5)"
    )

    parser.add_argument(
        "--n_estimators",
        type=int,
        default=100,
        help="Number of trees in forest (default: 100)"
    )

    parser.add_argument(
        "--max_depth",
        type=int,
        default=10,
        help="Maximum tree depth (default: 10)"
    )

    args = parser.parse_args()

    try:
        train_model(
            instrument_key=args.instrument_key,
            days=args.days,
            forward_days=args.forward_days,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth
        )
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
