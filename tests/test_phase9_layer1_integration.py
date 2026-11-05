from unittest.mock import patch

import pytest
from fastapi import HTTPException

from backend.app.ai import layer1


INSTRUMENT = "NSE_EQ|INE002A01018"


def rf_success(trend="bullish", confidence=0.80):
    return {
        "status": "success",
        "trend": trend,
        "confidence": confidence,
    }


def lstm_success(change=1.5, confidence=0.75, horizon=1):
    return {
        "status": "success",
        "predictions": [
            {
                "change_percent": change,
                "confidence": confidence,
                "horizon_sessions": horizon,
            }
        ],
    }


def test_both_models_successful():
    with (
        patch.object(layer1, "get_trend_prediction",
                     return_value=rf_success()),
        patch.object(layer1, "get_prediction",
                     return_value=lstm_success()),
    ):
        result = layer1.run_ai_signal(instrument_key=INSTRUMENT)

    assert result["status"] == "success"
    assert result["signal"] == "bullish"
    assert result["model_count"] == 2
    assert result["agreement"] == "2/2"
    assert result["agreement_type"] == "directional_consensus"
    assert result["confidence"] == pytest.approx(0.7775)
    assert result["source_outputs"]["random_forest"]["status"] == "success"
    assert result["source_outputs"]["lstm"]["status"] == "success"


def test_random_forest_only():
    with (
        patch.object(layer1, "get_trend_prediction",
                     return_value=rf_success("bearish", 0.80)),
        patch.object(layer1, "get_prediction",
                     return_value={"status": "error", "message": "LSTM unavailable"}),
    ):
        result = layer1.run_ai_signal(instrument_key=INSTRUMENT)

    assert result["signal"] == "bearish"
    assert result["model_count"] == 1
    assert result["agreement"] == "1/1"
    assert result["confidence"] == pytest.approx(0.80)


def test_lstm_only():
    with (
        patch.object(layer1, "get_trend_prediction",
                     return_value={"status": "error", "message": "RF unavailable"}),
        patch.object(layer1, "get_prediction",
                     return_value=lstm_success(-2.0, 0.70)),
    ):
        result = layer1.run_ai_signal(instrument_key=INSTRUMENT)

    assert result["signal"] == "bearish"
    assert result["model_count"] == 1
    assert result["agreement"] == "1/1"
    assert result["confidence"] == pytest.approx(0.70)


def test_both_models_unavailable_returns_503():
    with (
        patch.object(layer1, "get_trend_prediction",
                     return_value={"status": "error"}),
        patch.object(layer1, "get_prediction",
                     return_value={"status": "error"}),
    ):
        with pytest.raises(HTTPException) as exc:
            layer1.run_ai_signal(instrument_key=INSTRUMENT)

    assert exc.value.status_code == 503
    assert "No trained AI model" in exc.value.detail


def test_one_model_exception_does_not_hide_other_model():
    with (
        patch.object(layer1, "get_trend_prediction",
                     side_effect=RuntimeError("RF test failure")),
        patch.object(layer1, "get_prediction",
                     return_value=lstm_success(1.0, 0.65)),
    ):
        result = layer1.run_ai_signal(instrument_key=INSTRUMENT)

    assert result["status"] == "success"
    assert result["signal"] == "bullish"
    assert result["model_count"] == 1
    assert any(
        error["model"] == "RandomForest"
        for error in result["model_errors"]
    )


def test_both_model_exceptions_return_503():
    with (
        patch.object(layer1, "get_trend_prediction",
                     side_effect=RuntimeError("RF failure")),
        patch.object(layer1, "get_prediction",
                     side_effect=RuntimeError("LSTM failure")),
    ):
        with pytest.raises(HTTPException) as exc:
            layer1.run_ai_signal(instrument_key=INSTRUMENT)

    assert exc.value.status_code == 503


def test_blank_instrument_key_returns_400():
    with pytest.raises(HTTPException) as exc:
        layer1.run_ai_signal(instrument_key="   ")

    assert exc.value.status_code == 400


def test_conflicting_model_signals_are_reported():
    result = layer1._ensemble(
        rf_success("bullish", 0.90),
        lstm_success(-2.0, 0.40),
    )

    assert result["model_count"] == 2
    assert result["agreement"] == "1/2"
    assert result["agreement_type"] == "partial_directional_agreement"
    assert result["signal"] == "bullish"


def test_neutral_predictions_remain_neutral():
    result = layer1._ensemble(
        rf_success("neutral", 0.80),
        lstm_success(0.01, 0.70),
    )

    assert result["signal"] == "neutral"
    assert result["agreement"] == "2/2"
    assert result["agreement_type"] == "neutral_consensus"
    assert result["neutral_agreement_count"] == 2


def test_invalid_confidence_is_bounded():
    result = layer1._ensemble(
        rf_success("bullish", 1.50),
        lstm_success(1.0, -0.50),
    )

    for model in result["models"]:
        assert 0.0 <= model["confidence"] <= 1.0

    assert 0.0 <= result["confidence"] <= 1.0
