"""
Random Forest Trend Classification Model for StockAI

This module provides trend direction classification using Random Forest.
Features:
- Trend classification (bullish/bearish/neutral)
- Technical indicator-based features
- Confidence scores
- Model training and inference
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, List
import pickle
import os
from datetime import datetime, timedelta

# Keep NumPy/Pandas imports independent from scikit-learn.
# The previous combined try/except could set pandas to None when a
# scikit-learn import failed, which then crashed this module at class
# definition time on the pd.DataFrame type annotation.
import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
    SKLEARN_IMPORT_ERROR = None
except Exception as exc:
    SKLEARN_AVAILABLE = False
    SKLEARN_IMPORT_ERROR = exc
    RandomForestClassifier = None
    StandardScaler = None
    train_test_split = None


class RFTrendClassifier:
    """
    Random Forest-based trend classification model.

    Predicts trend direction based on technical indicators:
    - 0: Bearish (downtrend expected)
    - 1: Neutral (sideways movement)
    - 2: Bullish (uptrend expected)
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 10,
        model_path: Optional[str] = None,
    ):
        """
        Initialize RF trend classifier.

        Args:
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of trees
            model_path: Path to saved model directory
        """
        if not SKLEARN_AVAILABLE:
            raise RuntimeError(
                "scikit-learn is required for Random Forest predictions. "
                f"Import error: {SKLEARN_IMPORT_ERROR}. "
                "Install/repair it with: pip install -U scikit-learn"
            )

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        # Resolve the bundled model directory from this source file rather
        # than the process working directory. This keeps RF inference
        # working whether Uvicorn is started from the project root or
        # from the backend directory.
        default_model_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "models")
        )
        self.model_path = model_path or default_model_path

        # Scaler for feature normalization
        self.scaler = StandardScaler()

        # Model
        self.model: Optional[RandomForestClassifier] = None
        self.is_trained = False

        # Feature columns (removed sma_200 for faster training with less data)
        self.feature_columns = [
            'returns_1d', 'returns_5d', 'returns_10d',
            'sma_20', 'sma_50',
            'rsi', 'macd', 'macd_signal', 'macd_hist',
            'bb_position', 'atr_normalized',
            'volume_ratio', 'price_momentum',
            'volatility'
        ]

    def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators and features from OHLCV data."""
        df = df.copy()

        # Price returns
        df['returns_1d'] = df['close'].pct_change(1)
        df['returns_5d'] = df['close'].pct_change(5)
        df['returns_10d'] = df['close'].pct_change(10)

        # Simple Moving Averages
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()

        # Normalize SMAs (distance from current price)
        df['sma_20'] = (df['close'] - df['sma_20']) / df['close']
        df['sma_50'] = (df['close'] - df['sma_50']) / df['close']

        # RSI (Relative Strength Index)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # MACD (Moving Average Convergence Divergence)
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']

        # Normalize MACD
        df['macd'] = df['macd'] / df['close']
        df['macd_signal'] = df['macd_signal'] / df['close']
        df['macd_hist'] = df['macd_hist'] / df['close']

        # Bollinger Bands
        bb_middle = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        bb_upper = bb_middle + (bb_std * 2)
        bb_lower = bb_middle - (bb_std * 2)

        # BB position (where price is within bands)
        df['bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower)

        # ATR (Average True Range) - normalized
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        atr = true_range.rolling(14).mean()
        df['atr_normalized'] = atr / df['close']

        # Volume ratio
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(window=20).mean()

        # Price momentum
        df['price_momentum'] = df['close'] / df['close'].shift(10) - 1

        # Volatility (20-day rolling std of returns)
        df['volatility'] = df['returns_1d'].rolling(window=20).std()

        return df

    def _calculate_trend_labels(self, df: pd.DataFrame, forward_days: int = 5) -> pd.Series:
        """
        Calculate trend labels based on future price movement.

        Args:
            df: DataFrame with OHLCV data
            forward_days: Number of days to look ahead

        Returns:
            Series with trend labels (0=bearish, 1=neutral, 2=bullish)
        """
        # Calculate future returns
        future_returns = df['close'].shift(-forward_days) / df['close'] - 1

        # Define thresholds (can be adjusted)
        bullish_threshold = 0.02  # 2% gain
        bearish_threshold = -0.02  # 2% loss

        # Create labels
        labels = pd.Series(1, index=df.index)  # Default: neutral
        labels[future_returns > bullish_threshold] = 2  # Bullish
        labels[future_returns < bearish_threshold] = 0  # Bearish

        return labels

    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Prepare features from raw OHLCV data.

        Args:
            df: DataFrame with columns [open, high, low, close, volume, timestamp]

        Returns:
            Tuple of (features, labels) - labels is None for prediction
        """
        # Calculate indicators
        df = self._calculate_technical_indicators(df)

        # Drop NaN values
        df = df.dropna()

        if len(df) < 50:
            raise ValueError(f"Insufficient data: need at least 50 rows, got {len(df)}")

        # Extract features
        X = df[self.feature_columns].values

        return X, df

    def train(
        self,
        df: pd.DataFrame,
        forward_days: int = 5,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Dict:
        """
        Train the Random Forest model.

        Args:
            df: DataFrame with OHLCV data
            forward_days: Days ahead to predict trend
            test_size: Fraction of data for testing
            random_state: Random seed for reproducibility

        Returns:
            Dictionary with training metrics
        """
        print(f"Preparing features from {len(df)} candles...")

        # Prepare features and labels
        X, df_processed = self.prepare_features(df)
        y = self._calculate_trend_labels(df_processed, forward_days=forward_days)

        # Remove rows where we don't have future data
        valid_mask = y.notna()
        X = X[valid_mask]
        y = y[valid_mask].values

        print(f"Valid samples: {len(X)}")
        print(f"Class distribution: Bearish={np.sum(y==0)}, Neutral={np.sum(y==1)}, Bullish={np.sum(y==2)}")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # Fit scaler on training data
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train model
        print(f"Training Random Forest with {self.n_estimators} trees...")
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=random_state,
            n_jobs=-1,  # Use all CPU cores
            class_weight='balanced'  # Handle class imbalance
        )

        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True

        # Evaluate
        train_accuracy = self.model.score(X_train_scaled, y_train)
        test_accuracy = self.model.score(X_test_scaled, y_test)

        # Feature importance
        feature_importance = dict(zip(
            self.feature_columns,
            self.model.feature_importances_
        ))

        metrics = {
            'train_accuracy': float(train_accuracy),
            'test_accuracy': float(test_accuracy),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'feature_importance': feature_importance,
            'class_distribution': {
                'bearish': int(np.sum(y==0)),
                'neutral': int(np.sum(y==1)),
                'bullish': int(np.sum(y==2))
            }
        }

        print(f"\n✅ Training complete!")
        print(f"Train accuracy: {train_accuracy:.2%}")
        print(f"Test accuracy: {test_accuracy:.2%}")

        return metrics

    def predict_trend(self, df: pd.DataFrame) -> Dict:
        """
        Predict trend for the latest data point.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Dictionary with trend prediction and confidence
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first or load a trained model.")

        # Prepare features
        X, df_processed = self.prepare_features(df)

        # Use only the latest data point
        X_latest = X[-1:].reshape(1, -1)
        X_scaled = self.scaler.transform(X_latest)

        # Predict
        prediction = self.model.predict(X_scaled)[0]
        probabilities = self.model.predict_proba(X_scaled)[0]

        # Map prediction to label
        trend_map = {0: 'bearish', 1: 'neutral', 2: 'bullish'}
        trend = trend_map[prediction]
        confidence = float(probabilities[prediction])

        return {
            'trend': trend,
            'confidence': confidence,
            'probabilities': {
                'bearish': float(probabilities[0]),
                'neutral': float(probabilities[1]),
                'bullish': float(probabilities[2])
            },
            'prediction_class': int(prediction)
        }

    def save_model(self, filename: str = "rf_trend_model.pkl"):
        """Save trained model and scaler to disk."""
        if not self.is_trained:
            raise RuntimeError("No trained model to save")

        os.makedirs(self.model_path, exist_ok=True)

        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'trained_at': datetime.now().isoformat()
        }

        filepath = os.path.join(self.model_path, filename)
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"✅ Model saved to {filepath}")

    def load_model(self, filename: str = "rf_trend_model.pkl"):
        """Load trained model and scaler from disk."""
        filepath = os.path.join(self.model_path, filename)

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")

        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']
        self.n_estimators = model_data.get('n_estimators', self.n_estimators)
        self.max_depth = model_data.get('max_depth', self.max_depth)
        self.is_trained = True

        print(f"✅ Model loaded from {filepath}")
        print(f"   Trained at: {model_data.get('trained_at', 'unknown')}")
        return True
