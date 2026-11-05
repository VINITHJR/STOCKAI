"""
Advanced Automated Model Training and Management System
========================================================

Features:
- Automatic model training on startup if models are missing
- Model health checks and validation
- Automatic retraining triggers based on model age
- Parallel training of LSTM and Random Forest models
- Comprehensive error handling and recovery
- Training status monitoring and logging
- Model performance metrics tracking

Usage:
    python -m backend.app.ml.auto_train --instrument_key "NSE_EQ|INE002A01018"

Or import and use programmatically:
    from backend.app.ml.auto_train import ensure_models_trained, get_model_status
"""

import os
import sys
import time
import pickle
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

# Import training modules
from backend.app.services.upstox_service import get_historical_candles
from backend.app.ml.lstm_model import LSTMPricePredictor, TENSORFLOW_AVAILABLE
from backend.app.ml.rf_trend_classifier import RFTrendClassifier

# Paths
MODELS_DIR = Path("backend/app/ml/models")
STATUS_FILE = MODELS_DIR / "training_status.json"
LOG_FILE = MODELS_DIR / "training.log"


class TrainingLogger:
    """Simple logger for training operations."""

    def __init__(self, log_file: Path):
        self.log_file = log_file
        os.makedirs(log_file.parent, exist_ok=True)

    def log(self, message: str, level: str = "INFO"):
        """Write log message to file and console."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{level}] {message}"

        print(log_entry)

        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_entry + "\n")

    def info(self, message: str):
        self.log(message, "INFO")

    def error(self, message: str):
        self.log(message, "ERROR")

    def success(self, message: str):
        self.log(message, "SUCCESS")

    def warning(self, message: str):
        self.log(message, "WARNING")


logger = TrainingLogger(LOG_FILE)


def fetch_training_data(instrument_key: str, days: int = 365) -> list:
    """
    Fetch historical candle data for training.

    Args:
        instrument_key: Upstox instrument key
        days: Number of days of historical data

    Returns:
        List of candle dictionaries
    """
    logger.info(f"Fetching {days} days of data for {instrument_key}")

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

    # Convert to dict format
    candles = []
    for candle in raw_candles:
        if len(candle) < 6:
            continue

        candles.append({
            'timestamp': candle[0],
            'open': float(candle[1]),
            'high': float(candle[2]),
            'low': float(candle[3]),
            'close': float(candle[4]),
            'volume': float(candle[5]) if candle[5] else 0
        })

    logger.info(f"Fetched {len(candles)} candles successfully")
    return candles


def train_lstm_model(instrument_key: str, days: int = 365) -> Dict:
    """Refuse legacy single-instrument training for the shared LSTM artifact.

    Use ``python -m backend.app.ml.training_script --instrument-keys ...`` to
    train and publish the scale-aware v2 model across multiple instruments.
    The RandomForest training path in this module remains unchanged.
    """
    message = (
        "Shared LSTM training from a single instrument is disabled to prevent "
        "overwriting the multi-stock model. Run backend.app.ml.training_script "
        "with at least two --instrument-keys (preferably 8-20 instruments)."
    )
    logger.error(message)
    return {
        "status": "failed",
        "model": "LSTM",
        "error_code": "multi_stock_training_required",
        "error": message,
        "instrument_key": instrument_key,
        "timestamp": datetime.now().isoformat(),
    }

def train_rf_model(instrument_key: str, days: int = 365) -> Dict:
    """
    Train Random Forest trend classification model.

    Args:
        instrument_key: Upstox instrument key
        days: Days of historical data

    Returns:
        Training results dictionary
    """
    try:
        logger.info("="*60)
        logger.info("Starting Random Forest Model Training")
        logger.info("="*60)

        # Fetch data
        import pandas as pd
        candles = fetch_training_data(instrument_key, days)

        # Convert to DataFrame
        df = pd.DataFrame(candles)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)

        # Initialize classifier
        classifier = RFTrendClassifier(
            n_estimators=100,
            max_depth=10
        )

        # Train
        start_time = time.time()
        metrics = classifier.train(
            df=df,
            forward_days=5,
            test_size=0.2,
            random_state=42
        )
        training_time = time.time() - start_time

        # Save model
        classifier.save_model()

        # Test prediction
        prediction = classifier.predict_trend(df)

        logger.success("Random Forest Model Training Complete!")
        logger.info(f"Training time: {training_time:.2f}s")
        logger.info(f"Test accuracy: {metrics['test_accuracy']:.2%}")

        return {
            'status': 'success',
            'model': 'RandomForest',
            'training_time': training_time,
            'metrics': {
                'train_accuracy': metrics['train_accuracy'],
                'test_accuracy': metrics['test_accuracy'],
                'train_samples': metrics['train_samples'],
                'test_samples': metrics['test_samples']
            },
            'class_distribution': metrics['class_distribution'],
            'top_features': dict(sorted(
                metrics['feature_importance'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]),
            'test_prediction': prediction,
            'timestamp': datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Random Forest training failed: {e}")
        return {
            'status': 'failed',
            'model': 'RandomForest',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }


def train_all_models(instrument_key: str, days: int = 365, parallel: bool = True) -> Dict:
    """
    Train all models (LSTM and Random Forest).

    Args:
        instrument_key: Upstox instrument key
        days: Days of historical data
        parallel: Train models in parallel

    Returns:
        Combined training results
    """
    logger.info("🚀 Starting Complete Model Training Pipeline")
    logger.info(f"Instrument: {instrument_key}")
    logger.info(f"Historical days: {days}")
    logger.info(f"Parallel training: {parallel}")

    results = {
        'instrument_key': instrument_key,
        'training_days': days,
        'start_time': datetime.now().isoformat(),
        'models': {}
    }

    if parallel and TENSORFLOW_AVAILABLE:
        # Train models in parallel
        logger.info("Training models in parallel...")

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {
                executor.submit(train_lstm_model, instrument_key, days): 'LSTM',
                executor.submit(train_rf_model, instrument_key, days): 'RandomForest'
            }

            for future in as_completed(futures):
                model_name = futures[future]
                try:
                    result = future.result()
                    results['models'][model_name] = result

                    if result['status'] == 'success':
                        logger.success(f"✅ {model_name} trained successfully")
                    else:
                        logger.error(f"❌ {model_name} training failed")

                except Exception as e:
                    logger.error(f"❌ {model_name} training exception: {e}")
                    results['models'][model_name] = {
                        'status': 'failed',
                        'model': model_name,
                        'error': str(e)
                    }
    else:
        # Train models sequentially
        logger.info("Training models sequentially...")

        if TENSORFLOW_AVAILABLE:
            results['models']['LSTM'] = train_lstm_model(instrument_key, days)
        else:
            logger.warning("⚠️ Skipping LSTM (TensorFlow not available)")
            results['models']['LSTM'] = {
                'status': 'skipped',
                'model': 'LSTM',
                'reason': 'TensorFlow not available'
            }

        results['models']['RandomForest'] = train_rf_model(instrument_key, days)

    results['end_time'] = datetime.now().isoformat()
    results['success'] = all(
        m.get('status') in ['success', 'skipped']
        for m in results['models'].values()
    )

    # Save training status
    save_training_status(results)

    return results


def save_training_status(results: Dict):
    """Save training status to file."""
    os.makedirs(MODELS_DIR, exist_ok=True)

    with open(STATUS_FILE, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Training status saved to {STATUS_FILE}")


def load_training_status() -> Optional[Dict]:
    """Load training status from file."""
    if not STATUS_FILE.exists():
        return None

    try:
        with open(STATUS_FILE, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading training status: {e}")
        return None


def check_model_files() -> Dict[str, bool]:
    """
    Check if model files exist.

    Returns:
        Dictionary with model existence status
    """
    # Match the production LSTM v2 artifact filenames used by lstm_model.py.
    lstm_keras_v2 = MODELS_DIR / "lstm_model_v2.keras"
    lstm_scalers_v2 = MODELS_DIR / "lstm_scalers_v2.pkl"
    rf_model = MODELS_DIR / "rf_trend_model.pkl"

    return {
        'lstm': lstm_keras_v2.exists() and lstm_scalers_v2.exists(),
        'random_forest': rf_model.exists()
    }


def get_model_status() -> Dict:
    """
    Get comprehensive model status.

    Returns:
        Status dictionary with model health and training info
    """
    files_exist = check_model_files()
    training_status = load_training_status()

    status = {
        'models_exist': files_exist,
        'all_models_ready': all(files_exist.values()),
        'tensorflow_available': TENSORFLOW_AVAILABLE,
        'last_training': training_status,
        'recommendations': []
    }

    # Add recommendations
    if not files_exist['lstm']:
        status['recommendations'].append('Train LSTM model')

    if not files_exist['random_forest']:
        status['recommendations'].append('Train Random Forest model')

    if training_status:
        training_date = datetime.fromisoformat(training_status['start_time'])
        days_old = (datetime.now() - training_date).days

        if days_old > 30:
            status['recommendations'].append(
                f'Models are {days_old} days old. Consider retraining.'
            )

    return status


def ensure_models_trained(
    instrument_key: str = "NSE_EQ|INE002A01018",
    days: int = 365,
    force_retrain: bool = False
) -> Tuple[bool, Dict]:
    """
    Ensure models are trained and ready.

    This is the main function to call on application startup.

    Args:
        instrument_key: Default instrument for training
        days: Days of historical data
        force_retrain: Force retraining even if models exist

    Returns:
        (success, status_dict)
    """
    logger.info("🔍 Checking model status...")

    status = get_model_status()

    if status['all_models_ready'] and not force_retrain:
        logger.success("✅ All models are ready!")

        if status['last_training']:
            training_date = status['last_training']['start_time']
            logger.info(f"Last trained: {training_date}")

        return True, status

    # Train models
    logger.warning("⚠️ Models need training. Starting training pipeline...")

    if not status['tensorflow_available']:
        logger.warning(
            "⚠️ TensorFlow not available. "
            "LSTM predictions will not work. "
            "Install Python 3.11 or 3.12 with TensorFlow."
        )

    results = train_all_models(instrument_key, days, parallel=True)

    if results['success']:
        logger.success("🎉 All models trained successfully!")
        return True, results
    else:
        logger.error("❌ Some models failed to train")
        return False, results


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Advanced Model Training and Management System"
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
        "--force",
        action="store_true",
        help="Force retraining even if models exist"
    )

    parser.add_argument(
        "--status",
        action="store_true",
        help="Show model status only (no training)"
    )

    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Train models sequentially instead of parallel"
    )

    args = parser.parse_args()

    try:
        if args.status:
            # Show status only
            status = get_model_status()

            print("\n" + "="*60)
            print("MODEL STATUS")
            print("="*60)
            print(f"LSTM Model: {'✅ Ready' if status['models_exist']['lstm'] else '❌ Not Found'}")
            print(f"Random Forest: {'✅ Ready' if status['models_exist']['random_forest'] else '❌ Not Found'}")
            print(f"TensorFlow: {'✅ Available' if status['tensorflow_available'] else '❌ Not Available'}")

            if status['last_training']:
                print(f"\nLast Training: {status['last_training']['start_time']}")
                print(f"Instrument: {status['last_training']['instrument_key']}")

            if status['recommendations']:
                print("\nRecommendations:")
                for rec in status['recommendations']:
                    print(f"  • {rec}")

            print("="*60)

        else:
            # Train models
            success, results = ensure_models_trained(
                instrument_key=args.instrument_key,
                days=args.days,
                force_retrain=args.force
            )

            if success:
                print("\n" + "="*60)
                print("✅ TRAINING COMPLETE")
                print("="*60)
                print("All models are ready for predictions!")
                print(f"Status file: {STATUS_FILE}")
                print(f"Log file: {LOG_FILE}")
                print("="*60)
                sys.exit(0)
            else:
                print("\n" + "="*60)
                print("❌ TRAINING FAILED")
                print("="*60)
                print("Check the log file for details:")
                print(f"  {LOG_FILE}")
                print("="*60)
                sys.exit(1)

    except KeyboardInterrupt:
        logger.warning("\n⚠️ Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
