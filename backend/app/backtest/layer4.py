
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.app.backtest.cache import HistoricalDataCache
from backend.app.backtest.costs import calculate_costs
from backend.app.backtest.historical_data import fetch_historical_candles
from backend.app.backtest.metrics import calculate_metrics
from backend.app.backtest.models import ExecutionConfigRequest, Trade
from backend.app.ml.rf_trend_classifier import RFTrendClassifier


router = APIRouter(
    prefix="/api/backtest/ml",
    tags=["Phase 8 Layer 4 - ML Prediction Backtest"],
)


class MLPredictionBacktestRequest(BaseModel):
    """Request model for Phase 8 Layer 4 ML prediction replay."""

    instrument_key: str = Field(..., min_length=1)
    unit: str = Field("days", min_length=1)
    interval: str = Field("1", min_length=1)
    from_date: str = Field(..., min_length=10, max_length=10)
    to_date: str = Field(..., min_length=10, max_length=10)
    initial_capital: float = Field(100000.0, gt=0)
    quantity: int = Field(1, gt=0)
    forward_days: int = Field(5, ge=1, le=30)
    execution: ExecutionConfigRequest = Field(
        default_factory=ExecutionConfigRequest
    )


def _model_directory() -> Path:
    # backend/app/backtest/layer4.py -> backend/app/ml/models
    return Path(__file__).resolve().parents[1] / "ml" / "models"


def _load_rf_model() -> RFTrendClassifier:
    model = RFTrendClassifier(model_path=str(_model_directory()))
    model.load_model()
    if not model.is_trained or model.model is None:
        raise RuntimeError("Random Forest trend model is not available.")
    return model


def _prepare_predictions(
    candles: List[Dict[str, Any]],
    model: RFTrendClassifier,
) -> List[Dict[str, Any]]:
    if len(candles) < 50:
        raise ValueError(
            f"Insufficient historical candles for the ML model: "
            f"need at least 50, got {len(candles)}."
        )

    frame = pd.DataFrame(candles).copy()
    required = ["timestamp", "open", "high", "low", "close", "volume"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(
            "Historical candle data is missing required columns: "
            + ", ".join(missing)
        )

    for column in ["open", "high", "low", "close", "volume"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    feature_frame = model._calculate_technical_indicators(frame)
    feature_frame = feature_frame.dropna(
        subset=model.feature_columns
    ).copy()

    if feature_frame.empty:
        raise ValueError("No valid rows remain after calculating ML features.")

    X = feature_frame[model.feature_columns].values
    X_scaled = model.scaler.transform(X)

    predictions = model.model.predict(X_scaled)
    probabilities = model.model.predict_proba(X_scaled)
    classes = list(model.model.classes_)

    trend_map = {0: "bearish", 1: "neutral", 2: "bullish"}
    result: List[Dict[str, Any]] = []

    for row_index, (prediction, probability_row) in enumerate(
        zip(predictions, probabilities)
    ):
        class_value = int(prediction)
        probability_map = {
            trend_map.get(int(cls), str(int(cls))): round(
                float(probability_row[position]), 6
            )
            for position, cls in enumerate(classes)
        }
        confidence = float(
            probability_row[classes.index(prediction)]
        )

        result.append(
            {
                "timestamp": str(feature_frame.iloc[row_index]["timestamp"]),
                "trend": trend_map.get(class_value, "unknown"),
                "prediction_class": class_value,
                "confidence": round(confidence, 6),
                "probabilities": probability_map,
            }
        )

    return result


def _run_prediction_replay(
    candles: List[Dict[str, Any]],
    predictions: List[Dict[str, Any]],
    request: MLPredictionBacktestRequest,
) -> Dict[str, Any]:
    """
    Replay the already-trained RF model over historical candles.

    A prediction made at candle t is executed at candle t+1 open.
    Bullish -> BUY, bearish -> SELL, neutral -> HOLD.
    """

    prediction_by_timestamp = {
        item["timestamp"]: item for item in predictions
    }

    initial = float(request.initial_capital)
    cash = initial
    position: Dict[str, Any] | None = None
    trades: List[Trade] = []
    equity_curve: List[Dict[str, Any]] = []
    prediction_rows: List[Dict[str, Any]] = []
    total_execution_cost = 0.0

    for index, candle in enumerate(candles):
        timestamp = str(candle["timestamp"])
        close = float(candle["close"])
        prediction = prediction_by_timestamp.get(timestamp)

        if prediction is None:
            signal = "HOLD"
            trend = "unavailable"
            confidence = None
        else:
            trend = prediction["trend"]
            confidence = prediction["confidence"]
            signal = {
                "bullish": "BUY",
                "bearish": "SELL",
                "neutral": "HOLD",
            }.get(trend, "HOLD")

            prediction_rows.append(
                {
                    "timestamp": timestamp,
                    "trend": trend,
                    "signal": signal,
                    "confidence": confidence,
                    "probabilities": prediction["probabilities"],
                }
            )

        next_candle = candles[index + 1] if index + 1 < len(candles) else None

        if next_candle is not None:
            next_open = float(next_candle["open"])
            next_timestamp = str(next_candle["timestamp"])

            if position is None and signal == "BUY":
                notional = next_open * request.quantity
                entry_cost = calculate_costs(
                    next_open,
                    next_open,
                    request.quantity,
                    request.execution,
                ).buy_cost

                if cash >= notional + entry_cost:
                    cash -= notional + entry_cost
                    position = {
                        "entry_time": next_timestamp,
                        "entry_price": next_open,
                        "entry_cost": entry_cost,
                        "signal_time": timestamp,
                    }
                    total_execution_cost += entry_cost

            elif position is not None and signal == "SELL":
                entry = float(position["entry_price"])
                cost = calculate_costs(
                    entry,
                    next_open,
                    request.quantity,
                    request.execution,
                )
                gross = (next_open - entry) * request.quantity
                net = gross - cost.total_cost

                trade = Trade(
                    entry_time=str(position["entry_time"]),
                    exit_time=next_timestamp,
                    entry_price=entry,
                    exit_price=next_open,
                    quantity=request.quantity,
                    gross_pnl=round(gross, 4),
                    execution_cost=round(cost.total_cost, 4),
                    pnl=round(net, 4),
                    return_percent=round(
                        net / (entry * request.quantity) * 100
                        if entry > 0
                        else 0.0,
                        4,
                    ),
                )

                cash += next_open * request.quantity - cost.sell_cost
                total_execution_cost += cost.sell_cost
                trades.append(trade)
                position = None

        position_value = (
            close * request.quantity if position is not None else 0.0
        )
        equity_curve.append(
            {
                "timestamp": timestamp,
                "equity": round(cash + position_value, 4),
                "cash": round(cash, 4),
                "position_value": round(position_value, 4),
                "signal": signal,
                "trend": trend,
                "confidence": confidence,
            }
        )

    if position is not None:
        last = candles[-1]
        close = float(last["close"])
        entry = float(position["entry_price"])
        cost = calculate_costs(
            entry,
            close,
            request.quantity,
            request.execution,
        )
        gross = (close - entry) * request.quantity
        net = gross - cost.total_cost

        trade = Trade(
            entry_time=str(position["entry_time"]),
            exit_time=str(last["timestamp"]),
            entry_price=entry,
            exit_price=close,
            quantity=request.quantity,
            gross_pnl=round(gross, 4),
            execution_cost=round(cost.total_cost, 4),
            pnl=round(net, 4),
            return_percent=round(
                net / (entry * request.quantity) * 100
                if entry > 0
                else 0.0,
                4,
            ),
        )

        cash += close * request.quantity - cost.sell_cost
        total_execution_cost += cost.sell_cost
        trades.append(trade)

        if equity_curve:
            equity_curve[-1]["equity"] = round(cash, 4)
            equity_curve[-1]["cash"] = round(cash, 4)
            equity_curve[-1]["position_value"] = 0.0

    return {
        "initial_capital": initial,
        "final_capital": round(cash, 4),
        "total_pnl": round(cash - initial, 4),
        "total_return_percent": round(
            (cash - initial) / initial * 100,
            4,
        ),
        "total_trades": len(trades),
        "total_execution_cost": round(total_execution_cost, 4),
        "trades": [trade.model_dump() for trade in trades],
        "equity_curve": equity_curve,
        "predictions": prediction_rows,
    }


def _directional_accuracy(
    candles: List[Dict[str, Any]],
    predictions: List[Dict[str, Any]],
    forward_days: int,
) -> Dict[str, Any]:
    """Compare replayed trend classes with future returns.

    This is an evaluation statistic for the existing trained model; it is
    not used to generate the trading signal.
    """
    closes = [float(candle["close"]) for candle in candles]
    timestamps = [str(candle["timestamp"]) for candle in candles]
    index_by_timestamp = {
        timestamp: index
        for index, timestamp in enumerate(timestamps)
    }

    evaluated = 0
    correct = 0

    for prediction in predictions:
        index = index_by_timestamp.get(prediction["timestamp"])
        if index is None or index + forward_days >= len(closes):
            continue

        future_return = (
            closes[index + forward_days] / closes[index] - 1.0
        )
        actual = (
            "bullish"
            if future_return > 0.02
            else "bearish"
            if future_return < -0.02
            else "neutral"
        )

        evaluated += 1
        if actual == prediction["trend"]:
            correct += 1

    return {
        "forward_days": forward_days,
        "evaluated_predictions": evaluated,
        "correct_predictions": correct,
        "accuracy_percent": round(
            correct / evaluated * 100,
            4,
        ) if evaluated else None,
        "note": (
            "Replay accuracy uses future prices only for evaluation. "
            "It is not used to generate historical signals."
        ),
    }


@router.post("/run")
async def run_ml_prediction_backtest(
    request: MLPredictionBacktestRequest,
):
    try:
        cache = HistoricalDataCache()
        candles = cache.get(
            request.instrument_key,
            request.unit,
            request.interval,
            request.from_date,
            request.to_date,
        )
        cache_hit = candles is not None

        if candles is None:
            candles = await fetch_historical_candles(
                instrument_key=request.instrument_key,
                unit=request.unit,
                interval=request.interval,
                from_date=request.from_date,
                to_date=request.to_date,
            )

            if not candles:
                raise HTTPException(
                    status_code=404,
                    detail="No historical candles found.",
                )

            cache.set(
                request.instrument_key,
                request.unit,
                request.interval,
                request.from_date,
                request.to_date,
                candles,
            )

        model = _load_rf_model()
        predictions = _prepare_predictions(candles, model)
        replay = _run_prediction_replay(
            candles,
            predictions,
            request,
        )

        class TradeProxy:
            def __init__(self, payload: Dict[str, Any]):
                self.pnl = payload["pnl"]
                self.exit_time = payload["exit_time"]

        metrics = calculate_metrics(
            initial_capital=request.initial_capital,
            trades=[
                TradeProxy(payload)
                for payload in replay["trades"]
            ],
            equity_curve=replay["equity_curve"],
        )

        return {
            "success": True,
            "layer": "Phase 8 / Layer 4",
            "data_source": "cache" if cache_hit else "upstox",
            "model": {
                "name": "random_forest_trend_classifier",
                "prediction_horizon_days": request.forward_days,
                "model_path": str(_model_directory()),
                "trained_model_replay": True,
            },
            "candle_count": len(candles),
            "prediction_count": len(predictions),
            "prediction_evaluation": _directional_accuracy(
                candles,
                predictions,
                request.forward_days,
            ),
            "data": replay,
            "metrics": metrics,
            "execution_costs": request.execution.model_dump(),
        }

    except HTTPException:
        raise
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc
