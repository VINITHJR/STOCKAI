"""
Backend Startup Module
======================

Handles application initialization tasks:
- Model health checks
- Automatic model training if needed
- Service validation
- Configuration verification

This module runs on application startup to ensure all ML models
are ready before accepting prediction requests.
"""

import os
import sys
from pathlib import Path

# Add backend to path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from backend.app.ml.auto_train import (
    ensure_models_trained,
    get_model_status,
    logger
)


def initialize_ml_models(auto_train: bool = True) -> bool:
    """
    Initialize ML models on startup.

    Args:
        auto_train: If True, automatically train missing models

    Returns:
        True if models are ready
    """
    logger.info("=" * 60)
    logger.info("ML MODELS INITIALIZATION")
    logger.info("=" * 60)

    # Check model status
    status = get_model_status()

    logger.info(f"LSTM Model: {'✅ Ready' if status['models_exist']['lstm'] else '❌ Missing'}")
    logger.info(f"Random Forest: {'✅ Ready' if status['models_exist']['random_forest'] else '❌ Missing'}")
    logger.info(f"TensorFlow: {'✅ Available' if status['tensorflow_available'] else '⚠️ Not Available'}")

    if status['all_models_ready']:
        logger.success("All models are ready!")

        # Check model age
        if status['last_training']:
            from datetime import datetime
            training_date = datetime.fromisoformat(status['last_training']['start_time'])
            days_old = (datetime.now() - training_date).days

            if days_old > 0:
                logger.info(f"Models were trained {days_old} days ago")

        logger.info("=" * 60)
        return True

    # Models are missing
    if not auto_train:
        logger.warning("Models are missing. Set AUTO_TRAIN_MODELS=true to enable automatic training.")
        logger.info("=" * 60)
        return False

    # Auto-train models
    logger.warning("Models are missing. Starting automatic training...")
    logger.info("This may take 5-10 minutes...")

    try:
        # Use default Reliance stock for initial training
        # Users can retrain with their preferred instrument later
        success, results = ensure_models_trained(
            instrument_key="NSE_EQ|INE002A01018",  # Reliance
            days=365,
            force_retrain=False
        )

        if success:
            logger.success("🎉 Automatic training completed successfully!")
            logger.info("=" * 60)
            return True
        else:
            logger.error("Automatic training failed. Check logs for details.")
            logger.info("=" * 60)
            return False

    except Exception as e:
        logger.error(f"Automatic training error: {e}")
        logger.info("=" * 60)
        return False


def startup_checks():
    """
    Run all startup checks and initialization.

    This is the main function called when the backend starts.
    """
    print("\n" + "=" * 60)
    print("🚀 STOCKAI BACKEND STARTUP")
    print("=" * 60)

    # Check environment variables
    from dotenv import load_dotenv
    load_dotenv()

    required_env_vars = [
        'UPSTOX_API_KEY',
        'UPSTOX_API_SECRET',
        'UPSTOX_REDIRECT_URI'
    ]

    missing_vars = [var for var in required_env_vars if not os.getenv(var)]

    if missing_vars:
        print(f"⚠️  Missing environment variables: {', '.join(missing_vars)}")
        print("   Please check your .env file")
    else:
        print("✅ Environment variables loaded")

    # Initialize ML models
    auto_train = os.getenv('AUTO_TRAIN_MODELS', 'false').lower() == 'true'

    models_ready = initialize_ml_models(auto_train=auto_train)

    if not models_ready:
        print("\n⚠️  WARNING: ML models are not ready")
        print("   Predictions will not be available until models are trained")
        print("\n   To train models manually:")
        print("   python -m backend.app.ml.auto_train")
        print()

    print("=" * 60)
    print("✅ Backend startup complete")
    print("=" * 60)
    print()

    return models_ready


if __name__ == "__main__":
    # Can be run standalone to check startup status
    startup_checks()
