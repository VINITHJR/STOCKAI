"""Walk-forward evaluation and trading-usefulness backtest for StockAI LSTM v2.

Run from the StockAI project root:
python -m backend.app.ml.lstm_walk_forward --instrument-file backend/app/ml/models/lstm_instrument_keys.txt --days 1095 --epochs 30

This script trains temporary models for each chronological fold. It never calls
predictor.save() and never overwrites production LSTM artifacts.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from backend.app.services.upstox_service import get_historical_candles
from backend.app.ml.lstm_model import LSTMPricePredictor, TENSORFLOW_AVAILABLE, keras


HORIZONS = [1, 3, 7]
# Expanding training window; each test block is the next non-overlapping 10%:
# fold 1 tests 65-75%, fold 2 75-85%, fold 3 85-95%.
FOLDS = [
    {"name": "fold_1", "train_end": 0.55, "val_end": 0.65, "test_end": 0.75},
    {"name": "fold_2", "train_end": 0.65, "val_end": 0.75, "test_end": 0.85},
    {"name": "fold_3", "train_end": 0.75, "val_end": 0.85, "test_end": 0.95},
]
DEAD_BAND = 0.0005
COST_SENSITIVITY_BPS = [10, 30, 50]


def read_keys(instrument_keys: Sequence[str] | None, instrument_file: str | None) -> List[str]:
    keys = list(instrument_keys or [])
    if instrument_file:
        path = Path(instrument_file)
        if not path.exists():
            raise FileNotFoundError(f"Instrument-key file not found: {path}")
        for raw in path.read_text(encoding="utf-8-sig").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            # Supports generated lines like: NSE_EQ|ISIN  # SYMBOL
            key = line.split("#", 1)[0].strip()
            if key:
                keys.append(key)
    return list(dict.fromkeys(k for k in (str(x).strip() for x in keys) if k))


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
    output = []
    for row in raw:
        if not isinstance(row, (list, tuple)) or len(row) < 6:
            continue
        try:
            output.append({
                "timestamp": str(row[0]), "open": float(row[1]), "high": float(row[2]),
                "low": float(row[3]), "close": float(row[4]), "volume": float(row[5] or 0),
            })
        except (TypeError, ValueError):
            continue
    if not output:
        raise ValueError("No valid historical candles returned.")
    return output


def direction(value: float) -> str:
    if value > DEAD_BAND:
        return "bullish"
    if value < -DEAD_BAND:
        return "bearish"
    return "neutral"


def mean_abs(values) -> float | None:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.mean(np.abs(values))) if len(values) else None


def rmse(values) -> float | None:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.sqrt(np.mean(values ** 2))) if len(values) else None


def metric_block(rows: List[dict], horizon: int) -> dict:
    pred = np.asarray([r[f"pred_return_{horizon}d"] for r in rows], dtype=float)
    actual = np.asarray([r[f"actual_return_{horizon}d"] for r in rows], dtype=float)
    pred_price = np.asarray([r[f"pred_price_{horizon}d"] for r in rows], dtype=float)
    actual_price = np.asarray([r[f"actual_price_{horizon}d"] for r in rows], dtype=float)
    base = np.asarray([r["base_close"] for r in rows], dtype=float)
    err_pp = (pred - actual) * 100.0
    baseline_pp = (-actual) * 100.0
    actual_dirs = [direction(v) for v in actual]
    pred_dirs = [direction(v) for v in pred]
    correct = sum(a == p for a, p in zip(actual_dirs, pred_dirs))
    moving = [i for i, a in enumerate(actual_dirs) if a != "neutral"]
    moving_correct = sum(actual_dirs[i] == pred_dirs[i] for i in moving)
    return {
        "samples": int(len(rows)),
        "return_mae_pct_points": round(mean_abs(err_pp), 6) if len(rows) else None,
        "return_rmse_pct_points": round(rmse(err_pp), 6) if len(rows) else None,
        "price_mae_inr": round(mean_abs(pred_price - actual_price), 4) if len(rows) else None,
        "price_rmse_inr": round(rmse(pred_price - actual_price), 4) if len(rows) else None,
        "directional_accuracy_percent_including_neutral": round(100 * correct / len(rows), 3) if rows else None,
        "directional_accuracy_percent_actual_moves_only": round(100 * moving_correct / len(moving), 3) if moving else None,
        "baseline_no_change_return_mae_pct_points": round(mean_abs(baseline_pp), 6) if len(rows) else None,
        "baseline_no_change_price_mae_inr": round(mean_abs(base - actual_price), 4) if len(rows) else None,
        "beats_no_change_baseline_on_return_mae": bool(mean_abs(err_pp) < mean_abs(baseline_pp)) if rows else False,
    }


def make_samples(key: str, df, scaled: np.ndarray, seq_len: int, horizons: Sequence[int],
                 train_end: int, val_end: int, test_end: int):
    closes = df["close"].to_numpy(dtype=np.float64)
    max_h = max(horizons)
    train, val, test = [], [], []
    for base_idx in range(seq_len - 1, min(test_end - max_h, len(df) - max_h)):
        start = base_idx - seq_len + 1
        seq = scaled[start:base_idx + 1].astype(np.float32)
        if len(seq) != seq_len or closes[base_idx] <= 0:
            continue
        target_returns = {h: float(closes[base_idx + h] / closes[base_idx] - 1.0) for h in horizons}
        meta = {
            "instrument_key": key,
            "base_index": int(base_idx),
            "base_timestamp": str(df.iloc[base_idx]["timestamp"]),
            "base_close": float(closes[base_idx]),
            "entry_open_1d": float(df.iloc[base_idx + 1]["open"]),
            "exit_close_7d": float(df.iloc[base_idx + 7]["close"]) if base_idx + 7 < len(df) else None,
            "actual_returns": target_returns,
            "actual_prices": {h: float(closes[base_idx + h]) for h in horizons},
        }
        target_end = base_idx + max_h
        item = (seq, np.asarray([target_returns[h] for h in horizons], dtype=np.float32), meta)
        if target_end < train_end:
            train.append(item)
        elif base_idx >= train_end - 1 and target_end < val_end:
            val.append(item)
        elif base_idx >= val_end - 1 and target_end < test_end:
            test.append(item)
    return train, val, test


def max_drawdown_pct(returns: Sequence[float]) -> float:
    equity = 1.0
    peak = 1.0
    worst = 0.0
    for ret in returns:
        equity *= (1.0 + ret)
        peak = max(peak, equity)
        if peak > 0:
            worst = min(worst, equity / peak - 1.0)
    return 100.0 * worst


def trade_stats(returns: Sequence[float]) -> dict:
    arr = np.asarray(returns, dtype=float)
    arr = arr[np.isfinite(arr)]
    if not len(arr):
        return {
            "trade_count": 0, "win_rate_percent": None, "average_net_trade_return_percent": None,
            "compounded_return_percent": 0.0, "max_drawdown_percent": 0.0,
            "annualized_sharpe_7day_trades": None,
        }
    equity = float(np.prod(1.0 + arr))
    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    sharpe = (float(np.mean(arr)) / std * math.sqrt(252.0 / 7.0)) if std > 0 else None
    return {
        "trade_count": int(len(arr)),
        "win_rate_percent": round(100.0 * float(np.mean(arr > 0)), 3),
        "average_net_trade_return_percent": round(100.0 * float(np.mean(arr)), 4),
        "compounded_return_percent": round(100.0 * (equity - 1.0), 4),
        "max_drawdown_percent": round(max_drawdown_pct(arr), 4),
        "annualized_sharpe_7day_trades": round(sharpe, 4) if sharpe is not None else None,
    }


def build_trade_results(rows: List[dict], datasets: dict, edge_bps: float):
    """Long-or-cash, 7-session holding-period simulation with next-open entry.

    Each transaction-cost scenario is simulated independently so a signal taken
    in the 10-bps scenario cannot incorrectly block a trade in the 50-bps case.
    """
    by_stock_fold = {}
    for row in rows:
        by_stock_fold.setdefault((row["instrument_key"], row["fold"]), []).append(row)

    trades = []
    buy_hold_returns_by_stock = {}
    for (key, fold), fold_rows in by_stock_fold.items():
        fold_rows.sort(key=lambda r: r["base_index"])
        df = datasets[key]["df"]
        test_end = datasets[key]["fold_bounds"][fold]["test_end"]

        # Buy-and-hold benchmark for this fold: enter at first test session open,
        # exit at last test session close. Keep gross return; costs are applied
        # independently for each cost scenario below.
        first_base = fold_rows[0]["base_index"]
        last_exit_idx = min(test_end - 1, len(df) - 1)
        entry_idx = min(first_base + 1, len(df) - 1)
        if df.iloc[entry_idx]["open"] > 0:
            bh_gross = float(df.iloc[last_exit_idx]["close"] / df.iloc[entry_idx]["open"] - 1.0)
            buy_hold_returns_by_stock.setdefault(key, []).append((fold, bh_gross))

        for cost_bps in COST_SENSITIVITY_BPS:
            last_exit_index = -1
            cost_fraction = cost_bps / 10000.0
            threshold = (cost_bps + edge_bps) / 10000.0
            for row in fold_rows:
                base_idx = row["base_index"]
                exit_idx = base_idx + 7
                entry_idx = base_idx + 1
                if base_idx < last_exit_index or exit_idx >= test_end or entry_idx >= len(df):
                    continue
                predicted_close_to_close = row["pred_return_7d"]
                entry_price = float(df.iloc[entry_idx]["open"])
                exit_price = float(df.iloc[exit_idx]["close"])
                if not (entry_price > 0 and exit_price > 0 and row["base_close"] > 0):
                    continue

                # Model forecasts close(t) -> close(t+7), but execution enters at
                # open(t+1). Convert the predicted future close into an expected
                # entry-open-to-target-close return before applying the threshold.
                predicted_target_close = row["base_close"] * (1.0 + predicted_close_to_close)
                predicted_execution_return = predicted_target_close / entry_price - 1.0
                if not math.isfinite(predicted_execution_return) or predicted_execution_return <= threshold:
                    continue

                gross = exit_price / entry_price - 1.0
                net = (1.0 + gross) * (1.0 - cost_fraction) - 1.0
                trades.append({
                    "instrument_key": key, "fold": fold,
                    "base_timestamp": row["base_timestamp"],
                    "entry_timestamp": str(df.iloc[entry_idx]["timestamp"]),
                    "exit_timestamp": str(df.iloc[exit_idx]["timestamp"]),
                    "predicted_close_to_close_7d_return_percent": round(100.0 * predicted_close_to_close, 5),
                    "predicted_execution_return_percent": round(100.0 * predicted_execution_return, 5),
                    "entry_price": round(entry_price, 4),
                    "exit_price": round(exit_price, 4),
                    "gross_execution_return_percent": round(100.0 * gross, 5),
                    "round_trip_cost_bps": cost_bps,
                    "required_predicted_execution_edge_bps_including_margin": cost_bps + edge_bps,
                    "net_execution_return_percent": round(100.0 * net, 5),
                })
                # A held position blocks new entries until its exit date.
                last_exit_index = exit_idx

    report = {}
    keys = sorted({t["instrument_key"] for t in trades} | set(buy_hold_returns_by_stock.keys()))
    for key in keys:
        report[key] = {}
        for cost_bps in COST_SENSITIVITY_BPS:
            selected = [
                t for t in trades
                if t["instrument_key"] == key and t["round_trip_cost_bps"] == cost_bps
            ]
            selected.sort(key=lambda t: (t["base_timestamp"], t["fold"]))
            returns = [float(t["net_execution_return_percent"]) / 100.0 for t in selected]
            bh = buy_hold_returns_by_stock.get(key, [])
            bh.sort(key=lambda item: item[0])
            cost_fraction = cost_bps / 10000.0
            bh_net_returns = [(1.0 + gross) * (1.0 - cost_fraction) - 1.0 for _, gross in bh]
            bh_compounded = float(np.prod([1.0 + r for r in bh_net_returns]) - 1.0) if bh_net_returns else 0.0
            report[key][str(cost_bps)] = {
                "round_trip_cost_bps": cost_bps,
                "minimum_predicted_execution_return_threshold_percent": round((cost_bps + edge_bps) / 100.0, 4),
                **trade_stats(returns),
                "buy_and_hold_test_period_return_percent_after_round_trip_cost": round(100.0 * bh_compounded, 4),
                "cash_return_percent": 0.0,
                "interpretation_note": "Long-or-cash strategy; non-overlapping 7-session trades, next-session-open entry, 7th-session-close exit. Strategy and buy-and-hold both include the stated round-trip cost estimate per completed position.",
            }
    return trades, report


def run(keys: List[str], days: int, epochs: int, batch_size: int, edge_bps: float) -> dict:
    if not TENSORFLOW_AVAILABLE:
        raise RuntimeError("TensorFlow is unavailable in this Python environment.")
    if days < 365:
        raise ValueError("--days must be at least 365; use 1095 or more for this evaluation.")
    if epochs < 1 or batch_size < 1:
        raise ValueError("--epochs and --batch-size must both be positive.")
    if not math.isfinite(edge_bps) or edge_bps < 0:
        raise ValueError("--edge-bps must be a finite, non-negative number.")
    keys = list(dict.fromkeys(str(key).strip() for key in keys if str(key).strip()))
    if len(keys) < 2:
        raise ValueError("Provide at least two valid instrument keys. Use 8-20 stocks for a meaningful generalization test.")

    template = LSTMPricePredictor(sequence_length=60, prediction_horizons=HORIZONS)
    datasets = {}
    fetch_errors = {}
    for index, key in enumerate(keys, 1):
        print(f"[{index}/{len(keys)}] Fetching {days} calendar days for {key} ...")
        try:
            candles = fetch_candles(key, days)
            df = template._clean_candles(candles)
            if len(df) < 250:
                raise ValueError(f"Only {len(df)} usable rows after indicators; use more history.")
            datasets[key] = {
                "df": df,
                "raw": template._relative_features(df),
                "fold_bounds": {},
            }
            print(f"  Usable rows after indicators: {len(df)}")
        except Exception as exc:
            fetch_errors[key] = str(exc)
            print(f"  SKIP: {exc}")

    if len(datasets) < 2:
        raise RuntimeError(f"Fewer than two instruments have enough history. Errors: {fetch_errors}")

    all_predictions = []
    fold_summaries = []
    for fold in FOLDS:
        fold_name = fold["name"]
        print(f"\n=== {fold_name}: train to {fold['train_end']:.0%}, validation to {fold['val_end']:.0%}, test to {fold['test_end']:.0%} ===")
        fold_data = {}
        scaler_fit_rows = []
        for key, item in datasets.items():
            n = len(item["df"])
            train_end = int(n * fold["train_end"])
            val_end = int(n * fold["val_end"])
            test_end = int(n * fold["test_end"])
            item["fold_bounds"][fold_name] = {
                "train_end": train_end, "val_end": val_end, "test_end": test_end,
                "train_end_timestamp": str(item["df"].iloc[train_end - 1]["timestamp"]),
                "validation_end_timestamp": str(item["df"].iloc[val_end - 1]["timestamp"]),
                "test_end_timestamp": str(item["df"].iloc[test_end - 1]["timestamp"]),
            }
            fold_data[key] = (train_end, val_end, test_end)
            scaler_fit_rows.append(item["raw"][:train_end])

        # Clear graphs between folds and set a reproducible per-fold seed.
        # The test data remains untouched; only training order/model initialization changes.
        try:
            keras.backend.clear_session()
            keras.utils.set_random_seed(42 + len(fold_summaries))
        except Exception:
            pass
        predictor = LSTMPricePredictor(sequence_length=60, prediction_horizons=HORIZONS)
        predictor.feature_scaler.fit(np.concatenate(scaler_fit_rows, axis=0))
        train_samples, val_samples, test_samples = [], [], []
        usable_keys = []
        for key, item in datasets.items():
            train_end, val_end, test_end = fold_data[key]
            scaled = predictor.feature_scaler.transform(item["raw"])
            tr, va, te = make_samples(
                key, item["df"], scaled, predictor.sequence_length, HORIZONS,
                train_end, val_end, test_end,
            )
            if not tr or not va or not te:
                fetch_errors[f"{key}:{fold_name}"] = (
                    f"Insufficient samples: train={len(tr)}, validation={len(va)}, test={len(te)}."
                )
                continue
            usable_keys.append(key)
            train_samples.extend((x, y, {**m, "fold": fold_name}) for x, y, m in tr)
            val_samples.extend((x, y, {**m, "fold": fold_name}) for x, y, m in va)
            test_samples.extend((x, y, {**m, "fold": fold_name}) for x, y, m in te)

        if len(usable_keys) < 2 or not train_samples or not val_samples or not test_samples:
            raise RuntimeError(f"{fold_name} did not have enough usable samples for at least two stocks.")

        X_train = np.asarray([x for x, _, _ in train_samples], dtype=np.float32)
        y_train_raw = np.asarray([y for _, y, _ in train_samples], dtype=np.float64)
        X_val = np.asarray([x for x, _, _ in val_samples], dtype=np.float32)
        y_val_raw = np.asarray([y for _, y, _ in val_samples], dtype=np.float64)
        X_test = np.asarray([x for x, _, _ in test_samples], dtype=np.float32)

        # Returns at different horizons have different distributions. Fit a separate
        # target scaler for each horizon using TRAIN LABELS ONLY (never validation/test).
        # This prevents the largest-variance horizon from dominating the MSE objective.
        target_mean = np.mean(y_train_raw, axis=0)
        target_scale = np.std(y_train_raw, axis=0)
        target_scale = np.maximum(target_scale, 1e-4)
        y_train = ((y_train_raw - target_mean) / target_scale).astype(np.float32)
        y_val = ((y_val_raw - target_mean) / target_scale).astype(np.float32)

        predictor.model = predictor.build_model((predictor.sequence_length, len(predictor.feature_columns)))
        history = predictor.model.fit(
            X_train, y_train, validation_data=(X_val, y_val),
            epochs=epochs, batch_size=batch_size, shuffle=True, verbose=1,
            callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True)],
        )
        predicted_scaled = np.asarray(predictor.model.predict(X_test, verbose=0), dtype=np.float64)
        if predicted_scaled.ndim != 2 or predicted_scaled.shape != (len(X_test), len(HORIZONS)):
            raise RuntimeError(
                f"{fold_name} model returned unexpected prediction shape {predicted_scaled.shape}; "
                f"expected {(len(X_test), len(HORIZONS))}."
            )
        if not np.isfinite(predicted_scaled).all():
            raise RuntimeError(f"{fold_name} model produced NaN/inf standardized predictions.")
        # Invert target scaling before any price/return metric or trading calculation.
        predicted = predicted_scaled * target_scale[None, :] + target_mean[None, :]
        # Safety clip is applied in ORIGINAL return space, after inverse scaling.
        predicted = np.clip(predicted, -0.5, 0.5)
        fold_rows = []
        for index, (_, _, meta) in enumerate(test_samples):
            row = {
                "fold": meta["fold"],
                "instrument_key": meta["instrument_key"],
                "base_index": meta["base_index"],
                "base_timestamp": meta["base_timestamp"],
                "base_close": meta["base_close"],
                "entry_open_1d": meta["entry_open_1d"],
                "exit_close_7d": meta["exit_close_7d"],
            }
            for j, horizon in enumerate(HORIZONS):
                pred_ret = float(predicted[index][j])
                actual_ret = float(meta["actual_returns"][horizon])
                row[f"pred_return_{horizon}d"] = pred_ret
                row[f"actual_return_{horizon}d"] = actual_ret
                row[f"pred_price_{horizon}d"] = float(meta["base_close"] * (1 + pred_ret))
                row[f"actual_price_{horizon}d"] = float(meta["actual_prices"][horizon])
            fold_rows.append(row)
        all_predictions.extend(fold_rows)
        fold_summary = {
            "fold": fold_name,
            "stocks_evaluated": usable_keys,
            "training_samples": int(len(train_samples)),
            "validation_samples": int(len(val_samples)),
            "held_out_test_samples": int(len(test_samples)),
            "epochs_completed": int(len(history.history.get("loss", []))),
            "best_validation_loss_standardized_targets": float(min(history.history.get("val_loss", [math.nan]))),
            "target_scaling": {
                "method": "per-horizon z-score fitted on training labels only",
                "training_target_mean_percent": {
                    str(h): round(float(target_mean[j] * 100.0), 6)
                    for j, h in enumerate(HORIZONS)
                },
                "training_target_std_percent": {
                    str(h): round(float(target_scale[j] * 100.0), 6)
                    for j, h in enumerate(HORIZONS)
                },
            },
            "test_prediction_diagnostics": {
                str(h): {
                    "predicted_mean_percent": round(float(np.mean(predicted[:, j]) * 100.0), 6),
                    "predicted_std_percent": round(float(np.std(predicted[:, j]) * 100.0), 6),
                    "actual_mean_percent": round(float(np.mean([m["actual_returns"][h] for _, _, m in test_samples]) * 100.0), 6),
                    "actual_std_percent": round(float(np.std([m["actual_returns"][h] for _, _, m in test_samples]) * 100.0), 6),
                    "unique_rounded_predictions_5dp": int(len(set(np.round(predicted[:, j], 5)))),
                }
                for j, h in enumerate(HORIZONS)
            },
            "prediction_distribution_by_instrument": {
                key: {
                    str(h): {
                        "samples": int(sum(1 for r in fold_rows if r["instrument_key"] == key)),
                        "predicted_mean_percent": round(float(np.mean([r[f"pred_return_{h}d"] for r in fold_rows if r["instrument_key"] == key]) * 100.0), 6),
                        "predicted_std_percent": round(float(np.std([r[f"pred_return_{h}d"] for r in fold_rows if r["instrument_key"] == key]) * 100.0), 6),
                        "actual_std_percent": round(float(np.std([r[f"actual_return_{h}d"] for r in fold_rows if r["instrument_key"] == key]) * 100.0), 6),
                        "unique_rounded_predictions_5dp": int(len(set(np.round([r[f"pred_return_{h}d"] for r in fold_rows if r["instrument_key"] == key], 5)))),
                    }
                    for h in HORIZONS
                }
                for key in sorted({r["instrument_key"] for r in fold_rows})
            },
            "test_metrics": {str(h): metric_block(fold_rows, h) for h in HORIZONS},
            "test_date_ranges": {
                key: datasets[key]["fold_bounds"][fold_name]
                for key in usable_keys
            },
        }
        fold_summaries.append(fold_summary)
        print(f"{fold_name}: train={len(train_samples)}, validation={len(val_samples)}, test={len(test_samples)}")
        for h in HORIZONS:
            m = fold_summary["test_metrics"][str(h)]
            print(
                f"  {h}-day: return MAE={m['return_mae_pct_points']} pp; "
                f"direction accuracy={m['directional_accuracy_percent_including_neutral']}%; "
                f"beats no-change={m['beats_no_change_baseline_on_return_mae']}"
            )

    overall_metrics = {str(h): metric_block(all_predictions, h) for h in HORIZONS}
    trades, trading_report = build_trade_results(all_predictions, datasets, edge_bps)

    out_dir = Path("backend/app/ml/models")
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "lstm_v2_walk_forward_target_scaled_report.json"
    prediction_path = out_dir / "lstm_v2_walk_forward_target_scaled_predictions.csv"
    trades_path = out_dir / "lstm_v2_trading_trades_target_scaled.csv"
    report = {
        "status": "success",
        "method": "three expanding-window chronological folds; scaler fitted only on each fold's training rows",
        "production_artifacts_modified": False,
        "model_type": "temporary LSTM v2 retrained separately in each fold",
        "target_type": "future_return",
        "horizons_days": HORIZONS,
        "instruments_requested": keys,
        "instruments_loaded": list(datasets.keys()),
        "days_requested": int(days),
        "folds": fold_summaries,
        "overall_metrics_pooled_test_windows": overall_metrics,
        "evaluation_notes": [
            "Targets are standardized separately by horizon using training labels only; predictions are inverse-transformed before metrics.",
            "Compare every horizon against the no-change baseline; target scaling is an experiment, not proof of predictive skill.",
            "Pooled windows from different stocks/folds are not a single investable portfolio return.",
        ],
        "trading_simulation": {
            "strategy": "long-or-cash using predicted 7-session target close converted to expected next-session-open-to-target-close return; enter next session open, exit 7th session close; non-overlapping positions per instrument/fold",
            "extra_edge_margin_bps": float(edge_bps),
            "round_trip_cost_sensitivity_bps": COST_SENSITIVITY_BPS,
            "per_instrument_cost_sensitivity": trading_report,
            "trade_count_by_cost_bps": {
                str(cost): sum(1 for t in trades if t["round_trip_cost_bps"] == cost)
                for cost in COST_SENSITIVITY_BPS
            },
            "caveats": [
                "Trading costs are user-configurable estimates; actual Indian equity charges depend on product, broker, exchange, taxes and slippage.",
                "The forecast target is close-to-close; for trade selection it is converted to an expected next-open-to-target-close return using the observed entry open. The future close remains a model estimate, not a guaranteed execution price.",
                "No short selling, leverage, stop-loss, dividends, corporate actions or portfolio-level capital constraints are modeled.",
                "Small samples and overlapping forecast targets make results noisy; treat as research, not evidence of future profitability.",
            ],
        },
        "fetch_or_split_errors": fetch_errors,
        "report_path": str(report_path),
        "predictions_csv_path": str(prediction_path),
        "trades_csv_path": str(trades_path),
    }
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")

    if all_predictions:
        fields = list(all_predictions[0].keys())
        with prediction_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(all_predictions)
    trade_fields = list(trades[0].keys()) if trades else [
        "instrument_key", "fold", "base_timestamp", "entry_timestamp", "exit_timestamp",
        "predicted_close_to_close_7d_return_percent", "predicted_execution_return_percent",
        "entry_price", "exit_price", "gross_execution_return_percent", "round_trip_cost_bps",
        "required_predicted_execution_edge_bps_including_margin", "net_execution_return_percent",
    ]
    with trades_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=trade_fields)
        writer.writeheader()
        writer.writerows(trades)

    print("\nWALK-FORWARD EVALUATION COMPLETE")
    print(f"Report:      {report_path}")
    print(f"Predictions: {prediction_path}")
    print(f"Trades:      {trades_path}")
    print(json.dumps({
        "status": report["status"],
        "production_artifacts_modified": False,
        "overall_metrics_pooled_test_windows": overall_metrics,
        "trading_trade_count_by_cost_bps": report["trading_simulation"]["trade_count_by_cost_bps"],
        "report_path": str(report_path),
        "predictions_csv_path": str(prediction_path),
        "trades_csv_path": str(trades_path),
        "fetch_or_split_errors": fetch_errors,
    }, indent=2, allow_nan=False))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward validation and trading-usefulness test for StockAI LSTM v2.")
    parser.add_argument("--instrument-keys", nargs="*", default=[], help="Optional Upstox instrument keys.")
    parser.add_argument("--instrument-file", default=None, help="Text file with one Upstox instrument key per line; comments after # are allowed.")
    parser.add_argument("--days", type=int, default=1095, help="Calendar days of historical data per stock; 1095 is recommended.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--edge-bps", type=float, default=50.0, help="Extra predicted edge required beyond estimated round-trip cost, in basis points.")
    args = parser.parse_args()
    keys = read_keys(args.instrument_keys, args.instrument_file)
    run(keys, days=args.days, epochs=args.epochs, batch_size=args.batch_size, edge_bps=args.edge_bps)


if __name__ == "__main__":
    main()
