"""Scale-aware, multi-stock LSTM forecaster for StockAI.

Version 2 predicts future returns (not absolute prices) and uses relative
technical features so that one model can be trained across differently priced
stocks. It deliberately loads/saves v2 artifacts separately from the old H5
model, because old weights predict absolute prices and are not compatible with
this target definition.
"""
from __future__ import annotations

import os
import pickle
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

try:
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover
    np = None
    pd = None
    StandardScaler = None
    SKLEARN_AVAILABLE = False

try:
    from tensorflow import keras
    from tensorflow.keras import layers
    TENSORFLOW_AVAILABLE = True
except Exception:  # pragma: no cover - platform dependent
    keras = None
    layers = None
    TENSORFLOW_AVAILABLE = False


class LSTMPricePredictor:
    MODEL_VERSION = 2

    def __init__(
        self,
        sequence_length: int = 60,
        prediction_horizons: Optional[List[int]] = None,
        model_path: Optional[str] = None,
    ) -> None:
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("Install numpy, pandas, and scikit-learn to use the LSTM model.")
        if not TENSORFLOW_AVAILABLE:
            raise RuntimeError("TensorFlow is unavailable in this Python environment; LSTM cannot load or train.")
        self.sequence_length = int(sequence_length)
        self.prediction_horizons = sorted(set(int(x) for x in (prediction_horizons or [1, 3, 7])))
        if self.sequence_length < 10 or not self.prediction_horizons or min(self.prediction_horizons) < 1:
            raise ValueError("sequence_length must be >= 10 and horizons must be positive integers.")
        self.model_path = model_path or "backend/app/ml/models"
        self.model_file = os.path.join(self.model_path, "lstm_model_v2.keras")
        self.scaler_file = os.path.join(self.model_path, "lstm_scalers_v2.pkl")
        self.feature_scaler = StandardScaler()
        self.model = None
        self.is_trained = False
        # Names are retained for predictable model metadata, but these values
        # are now relative/scale-invariant features, not raw price levels.
        self.feature_columns = [
            "open", "high", "low", "close", "volume", "sma_20", "sma_50",
            "rsi", "macd", "macd_signal", "bb_upper", "bb_middle", "bb_lower", "atr"
        ]

    def _calculate_technical_indicators(self, df: "pd.DataFrame") -> "pd.DataFrame":
        df = df.copy()
        close = df["close"]
        df["sma_20"] = close.rolling(20, min_periods=20).mean()
        df["sma_50"] = close.rolling(50, min_periods=50).mean()
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14, min_periods=14).mean()
        loss = (-delta.clip(upper=0)).rolling(14, min_periods=14).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        # A rising window with no losses has RSI=100; a fully flat window is 50.
        rsi = rsi.mask((loss == 0) & (gain > 0), 100.0)
        rsi = rsi.mask((loss == 0) & (gain == 0), 50.0)
        df["rsi"] = rsi.fillna(50.0)
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        df["macd"] = ema12 - ema26
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["bb_middle"] = close.rolling(20, min_periods=20).mean()
        std = close.rolling(20, min_periods=20).std()
        df["bb_upper"] = df["bb_middle"] + 2 * std
        df["bb_lower"] = df["bb_middle"] - 2 * std
        prev_close = close.shift(1)
        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ], axis=1).max(axis=1)
        df["atr"] = tr.rolling(14, min_periods=14).mean()
        return df

    def _clean_candles(self, candles: Sequence[Dict]) -> "pd.DataFrame":
        if not candles:
            raise ValueError("No candles supplied.")
        df = pd.DataFrame(list(candles)).copy()
        required = ["timestamp", "open", "high", "low", "close", "volume"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Candles are missing required fields: {missing}")
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=required).sort_values("timestamp")
        df = df.drop_duplicates(subset=["timestamp"], keep="last")
        # Discard impossible prices instead of allowing corrupted candles into training.
        valid_prices = (
            (df["open"] > 0) & (df["high"] > 0) & (df["low"] > 0) &
            (df["close"] > 0) & (df["volume"] >= 0) &
            (df["high"] >= df[["open", "low", "close"]].max(axis=1)) &
            (df["low"] <= df[["open", "high", "close"]].min(axis=1))
        )
        df = df.loc[valid_prices].copy()
        if len(df) < self.sequence_length + max(self.prediction_horizons) + 50:
            raise ValueError(
                f"Need at least {self.sequence_length + max(self.prediction_horizons) + 50} valid candles; got {len(df)}."
            )
        df = self._calculate_technical_indicators(df)
        df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
        return df

    def _relative_features(self, df: "pd.DataFrame") -> "np.ndarray":
        close = df["close"].astype(float).replace(0, np.nan)
        # Price-derived features are ratios to close; volume uses log scale.
        features = pd.DataFrame(index=df.index)
        features["open"] = df["open"] / close
        features["high"] = df["high"] / close
        features["low"] = df["low"] / close
        features["close"] = close.pct_change().fillna(0.0)
        features["volume"] = np.log1p(df["volume"].clip(lower=0))
        features["sma_20"] = df["sma_20"] / close
        features["sma_50"] = df["sma_50"] / close
        features["rsi"] = df["rsi"] / 100.0
        features["macd"] = df["macd"] / close
        features["macd_signal"] = df["macd_signal"] / close
        features["bb_upper"] = df["bb_upper"] / close
        features["bb_middle"] = df["bb_middle"] / close
        features["bb_lower"] = df["bb_lower"] / close
        features["atr"] = df["atr"] / close
        return features.replace([np.inf, -np.inf], np.nan).fillna(0.0)[self.feature_columns].to_numpy(dtype=np.float64)

    def _samples(self, df: "pd.DataFrame", features: "np.ndarray") -> Tuple[list, list]:
        X, y = [], []
        closes = df["close"].to_numpy(dtype=np.float64)
        max_horizon = max(self.prediction_horizons)
        count = len(df) - self.sequence_length - max_horizon + 1
        for i in range(max(0, count)):
            end = i + self.sequence_length
            base = closes[end - 1]
            if base <= 0:
                continue
            targets = [closes[end + h - 1] / base - 1.0 for h in self.prediction_horizons]
            X.append(features[i:end])
            y.append(targets)
        return X, y

    def _samples_with_target_end(self, df: "pd.DataFrame", features: "np.ndarray"):
        """Return windows, future-return targets, and each window's last target row."""
        X, y, target_ends = [], [], []
        closes = df["close"].to_numpy(dtype=np.float64)
        max_horizon = max(self.prediction_horizons)
        count = len(df) - self.sequence_length - max_horizon + 1
        for i in range(max(0, count)):
            end = i + self.sequence_length
            base = closes[end - 1]
            if not np.isfinite(base) or base <= 0:
                continue
            targets = [closes[end + h - 1] / base - 1.0 for h in self.prediction_horizons]
            if not np.isfinite(targets).all():
                continue
            X.append(features[i:end])
            y.append(targets)
            target_ends.append(end + max_horizon - 1)
        return X, y, target_ends

    def prepare_data(self, candles: List[Dict], for_training: bool = True):
        """Compatibility helper; targets are v2 future returns, never absolute prices.

        When preparing training data, fit the scaler on the earlier 80% of rows
        only. Callers should still split the returned windows chronologically.
        """
        df = self._clean_candles(candles)
        raw_features = self._relative_features(df)
        if for_training:
            fit_end = max(self.sequence_length + max(self.prediction_horizons) + 1, int(len(df) * 0.8))
            fit_end = min(fit_end, len(df))
            self.feature_scaler.fit(raw_features[:fit_end])
            scaled = self.feature_scaler.transform(raw_features)
            X, y, _ = self._samples_with_target_end(df, scaled)
            if not X:
                return None
            return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)
        if not hasattr(self.feature_scaler, "n_features_in_"):
            raise RuntimeError("V2 feature scaler is not fitted. Train the v2 multi-stock model first.")
        scaled = self.feature_scaler.transform(raw_features)
        return np.asarray(scaled[-self.sequence_length:][None, ...], dtype=np.float32), None

    def build_model(self, input_shape: Tuple[int, int]):
        model = keras.Sequential([
            keras.Input(shape=input_shape),
            layers.LSTM(96, return_sequences=True),
            layers.Dropout(0.2),
            layers.LSTM(48, return_sequences=True),
            layers.Dropout(0.2),
            layers.LSTM(24),
            layers.Dropout(0.15),
            layers.Dense(24, activation="relu"),
            layers.Dense(len(self.prediction_horizons), activation="linear"),
        ])
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0005), loss="mse", metrics=["mae"])
        return model

    def train_multi_stock(
        self,
        datasets: Sequence[Dict],
        epochs: int = 40,
        batch_size: int = 32,
        validation_fraction: float = 0.2,
        verbose: int = 1,
    ) -> Dict:
        """Train one shared model on multiple instruments.

        datasets items: {"instrument_key": str, "candles": list[dict]}.
        Validation windows are the latest windows per instrument to reduce
        chronological leakage. Use multiple liquid stocks for generalization.
        """
        if len(datasets) < 2:
            raise ValueError("Provide at least 2 different instruments. For a useful shared model, use 8-20 stocks if possible.")
        prepared = []
        scaler_fit_rows = []
        for item in datasets:
            key = str(item.get("instrument_key") or "unknown")
            df = self._clean_candles(item.get("candles") or [])
            raw = self._relative_features(df)
            split_row = max(self.sequence_length + max(self.prediction_horizons) + 1, int(len(df) * (1 - validation_fraction)))
            scaler_fit_rows.append(raw[:split_row])
            prepared.append((key, df, raw, split_row))
        self.feature_scaler.fit(np.concatenate(scaler_fit_rows, axis=0))

        train_X, train_y, val_X, val_y = [], [], [], []
        per_stock = {}
        for key, df, raw, split_row in prepared:
            scaled = self.feature_scaler.transform(raw)
            closes = df["close"].to_numpy(dtype=np.float64)
            stock_train = 0
            stock_val = 0
            max_horizon = max(self.prediction_horizons)
            sample_count = len(df) - self.sequence_length - max_horizon + 1
            for i in range(max(0, sample_count)):
                end = i + self.sequence_length
                target_end = end + max_horizon - 1
                base = closes[end - 1]
                if base <= 0:
                    continue
                target = [closes[end + h - 1] / base - 1.0 for h in self.prediction_horizons]
                seq = scaled[i:end].astype(np.float32)
                # The target date must be on the corresponding side of split.
                if target_end < split_row:
                    train_X.append(seq); train_y.append(target); stock_train += 1
                elif i >= max(0, split_row - self.sequence_length):
                    val_X.append(seq); val_y.append(target); stock_val += 1
            per_stock[key] = {"training_samples": stock_train, "validation_samples": stock_val, "candles": len(df)}
        if not train_X or not val_X:
            raise ValueError("Not enough multi-stock windows for chronological train/validation split; increase --days or provide more history.")
        X_train = np.asarray(train_X, dtype=np.float32)
        y_train = np.asarray(train_y, dtype=np.float32)
        X_val = np.asarray(val_X, dtype=np.float32)
        y_val = np.asarray(val_y, dtype=np.float32)
        try:
            keras.utils.set_random_seed(42)
        except Exception:
            pass
        self.model = self.build_model((self.sequence_length, len(self.feature_columns)))
        callbacks = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True)]
        # Each sample is an independent historical window; shuffling window order
        # improves mixed-stock batches and does not shuffle data across time inside a window.
        hist = self.model.fit(
            X_train, y_train, validation_data=(X_val, y_val), epochs=int(epochs),
            batch_size=int(batch_size), shuffle=True, verbose=verbose, callbacks=callbacks,
        )
        self.is_trained = True
        val_pred = np.asarray(self.model.predict(X_val, verbose=0), dtype=np.float64)
        validation_baseline_metrics = {}
        for column, horizon in enumerate(self.prediction_horizons):
            model_mae = float(np.mean(np.abs(val_pred[:, column] - y_val[:, column])))
            no_change_mae = float(np.mean(np.abs(y_val[:, column])))
            validation_baseline_metrics[str(horizon)] = {
                "samples": int(len(y_val)),
                "model_return_mae_percentage_points": round(model_mae * 100.0, 6),
                "no_change_return_mae_percentage_points": round(no_change_mae * 100.0, 6),
                "beats_no_change_baseline": bool(model_mae < no_change_mae),
            }
        self.save()
        return {
            "status": "success", "model_version": self.MODEL_VERSION,
            "validation_baseline_metrics": validation_baseline_metrics,
            "validation_metrics_note": "Diagnostic only: these windows also drive early stopping; use the separate walk-forward test before promoting a candidate.",
            "instruments": len(prepared), "per_instrument": per_stock,
            "training_samples": int(len(X_train)), "validation_samples": int(len(X_val)),
            "epochs_completed": len(hist.history.get("loss", [])),
            "loss": hist.history.get("loss", []), "val_loss": hist.history.get("val_loss", []),
            "mae": hist.history.get("mae", []), "val_mae": hist.history.get("val_mae", []),
            "model_file": self.model_file, "scaler_file": self.scaler_file,
        }

    def train(self, candles: List[Dict], epochs: int = 40, batch_size: int = 32, validation_split: float = 0.2) -> Dict:
        """Single-stock compatibility path with chronological, target-safe splits.

        This path is for smoke testing only and never saves the shared production
        artifacts. Use train_multi_stock() for production training.
        """
        if not candles:
            raise ValueError("No candles supplied.")
        if not 0.05 <= float(validation_split) <= 0.4:
            raise ValueError("validation_split must be between 0.05 and 0.4.")
        df = self._clean_candles(candles)
        raw = self._relative_features(df)
        split_row = int(len(df) * (1.0 - float(validation_split)))
        split_row = max(self.sequence_length + max(self.prediction_horizons) + 1, split_row)
        if split_row >= len(df) - max(self.prediction_horizons):
            raise ValueError("Not enough candles for a chronological train/validation split.")
        self.feature_scaler.fit(raw[:split_row])
        scaled = self.feature_scaler.transform(raw)
        X, y, target_ends = self._samples_with_target_end(df, scaled)
        train_idx, val_idx = [], []
        validation_start = max(0, split_row - self.sequence_length)
        for idx, (i, target_end) in enumerate(zip(range(len(X)), target_ends)):
            # Training labels must end strictly before the split. Validation
            # labels must be entirely in the validation period; their input
            # sequence may use earlier historical context, as it should live.
            if target_end < split_row:
                train_idx.append(idx)
            else:
                # Recover the original window start from the sequential sample index.
                window_start = i
                if window_start >= validation_start:
                    val_idx.append(idx)
        if not train_idx or not val_idx:
            raise ValueError("Not enough samples for leakage-safe chronological training and validation.")
        X_train = np.asarray([X[i] for i in train_idx], dtype=np.float32)
        y_train = np.asarray([y[i] for i in train_idx], dtype=np.float32)
        X_val = np.asarray([X[i] for i in val_idx], dtype=np.float32)
        y_val = np.asarray([y[i] for i in val_idx], dtype=np.float32)
        try:
            keras.utils.set_random_seed(42)
        except Exception:
            pass
        self.model = self.build_model((self.sequence_length, len(self.feature_columns)))
        hist = self.model.fit(
            X_train, y_train, validation_data=(X_val, y_val),
            epochs=int(epochs), batch_size=int(batch_size), shuffle=True, verbose=1,
            callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True)],
        )
        self.is_trained = True
        return {
            "loss": hist.history.get("loss", []), "val_loss": hist.history.get("val_loss", []),
            "mae": hist.history.get("mae", []), "val_mae": hist.history.get("val_mae", []),
            "epochs": len(hist.history.get("loss", [])), "training_samples": len(train_idx),
            "validation_samples": len(val_idx), "model_version": self.MODEL_VERSION,
            "saved": False,
            "warning": "Single-instrument training is for smoke testing only and was not saved; use train_multi_stock for the shared production model.",
        }

    def predict(self, candles: List[Dict], current_price: float) -> Dict:
        if self.model is None or not self.is_trained:
            raise RuntimeError("V2 model not loaded/trained. Train the v2 multi-stock model first.")
        try:
            current_price = float(current_price)
        except (TypeError, ValueError):
            return {"status": "error", "error_code": "invalid_current_price", "message": "current_price must be numeric."}
        if not np.isfinite(current_price) or current_price <= 0:
            return {"status": "error", "error_code": "invalid_current_price", "message": "current_price must be finite and positive."}
        try:
            cleaned = self._clean_candles(candles)
            raw_features = self._relative_features(cleaned)
            if not hasattr(self.feature_scaler, "n_features_in_"):
                return {"status": "error", "error_code": "scaler_not_fitted", "message": "The saved v2 feature scaler is not fitted."}
            scaled = self.feature_scaler.transform(raw_features)
            X = np.asarray(scaled[-self.sequence_length:][None, ...], dtype=np.float32)
            predicted_returns = np.asarray(self.model.predict(X, verbose=0), dtype=np.float64)
            if predicted_returns.shape != (1, len(self.prediction_horizons)) or not np.isfinite(predicted_returns).all():
                return {"status": "error", "error_code": "invalid_model_output", "message": "Model output shape or values are invalid."}
            # Avoid presenting mathematically extreme outputs as sensible prices.
            predicted_returns = np.clip(predicted_returns[0], -0.5, 0.5)
            recent = cleaned["close"].tail(30).to_numpy(dtype=float)
            recent = recent[np.isfinite(recent) & (recent > 0)]
            if len(recent) < 2:
                return {"status": "error", "error_code": "insufficient_recent_prices", "message": "Not enough valid recent closes."}
            volatility = float(np.std(np.diff(np.log(recent))))
            results = []
            for horizon, ret in zip(self.prediction_horizons, predicted_returns):
                predicted_price = current_price * (1.0 + float(ret))
                change = float(ret) * 100.0
                uncertainty = current_price * min(0.5, volatility * np.sqrt(horizon) * 2.0)
                results.append({
                    # Legacy field retained for frontend compatibility; these are trading sessions.
                    "horizon_days": int(horizon), "horizon_sessions": int(horizon),
                    "predicted_price": round(predicted_price, 2),
                    "change_percent": round(change, 2),
                    # This is a heuristic score, not a calibrated probability.
                    "confidence": round(float(max(0.0, min(0.75, 0.65 - volatility * 5.0))), 2),
                    "lower_bound": round(max(0.01, predicted_price - uncertainty), 2),
                    "upper_bound": round(predicted_price + uncertainty, 2),
                    "direction": "bullish" if change > 0.05 else "bearish" if change < -0.05 else "neutral",
                })
            return {"status": "success", "model_version": self.MODEL_VERSION, "current_price": current_price,
                    "predictions": results, "model_type": "LSTM", "target_type": "future_return",
                    "confidence_method": "uncalibrated_volatility_heuristic", "timestamp": datetime.now().isoformat()}
        except (ValueError, RuntimeError, KeyError, TypeError) as exc:
            return {"status": "error", "error_code": "prediction_failed", "message": str(exc)}

    def save(self) -> None:
        if self.model is None or not self.is_trained:
            raise RuntimeError("Cannot save an untrained model.")
        os.makedirs(self.model_path, exist_ok=True)
        self.model.save(self.model_file)
        payload = {
            "model_version": self.MODEL_VERSION,
            "target_type": "future_return",
            "feature_scaler": self.feature_scaler,
            "sequence_length": self.sequence_length,
            "prediction_horizons": self.prediction_horizons,
            "feature_columns": self.feature_columns,
        }
        tmp = self.scaler_file + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, self.scaler_file)

    def load(self) -> bool:
        """Load only v2 artifacts; never silently load incompatible legacy absolute-price weights."""
        if not (os.path.isfile(self.model_file) and os.path.isfile(self.scaler_file)):
            return False
        try:
            with open(self.scaler_file, "rb") as f:
                payload = pickle.load(f)
            if payload.get("model_version") != self.MODEL_VERSION or payload.get("target_type") != "future_return":
                raise ValueError("Saved LSTM artifacts are not v2 future-return artifacts; retrain with training_script.py.")
            self.sequence_length = int(payload["sequence_length"])
            self.prediction_horizons = [int(x) for x in payload["prediction_horizons"]]
            self.feature_columns = list(payload["feature_columns"])
            self.feature_scaler = payload["feature_scaler"]
            self.model = keras.models.load_model(self.model_file, compile=False)
            self.is_trained = True
            return True
        except Exception as exc:
            print(f"Could not load StockAI LSTM v2 artifacts: {exc}")
            self.model = None
            self.is_trained = False
            return False
