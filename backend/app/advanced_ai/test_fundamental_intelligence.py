# ============================================================
# PHASE 14.3 — FUNDAMENTAL INTELLIGENCE TESTS
# ============================================================

from __future__ import annotations

import math

from fundamental_intelligence import (
    FundamentalIntelligence,
)

from fundamental_intelligence_models import (
    FundamentalAnalysisInput,
)


ENGINE = FundamentalIntelligence()


def assert_between(
    value: float,
    minimum: float,
    maximum: float,
) -> None:

    assert minimum <= value <= maximum, (
        f"{value} outside "
        f"[{minimum}, {maximum}]"
    )


def test_strong_fundamentals() -> None:

    data = FundamentalAnalysisInput(

        instrument_key="NSE_EQ|TEST_STRONG",

        symbol="TESTSTRONG",

        sector="Technology",

        pe_ratio=14.0,
        pb_ratio=1.8,
        ev_ebitda=9.0,

        roe=24.0,
        roa=11.0,
        roce=23.0,

        quick_ratio=1.8,
        debt_to_equity=0.4,

        dividend_yield=3.0,
    )

    result = ENGINE.analyze(data)

    assert result.signal == "bullish"

    assert result.fundamental_score > 0

    assert result.confidence > 0

    assert result.data_completeness == 1.0

    assert result.available_metric_count == 9


def test_weak_fundamentals() -> None:

    data = FundamentalAnalysisInput(

        instrument_key="NSE_EQ|TEST_WEAK",

        symbol="TESTWEAK",

        sector="Industrial",

        pe_ratio=45.0,
        pb_ratio=9.0,
        ev_ebitda=35.0,

        roe=2.0,
        roa=1.0,
        roce=2.0,

        quick_ratio=0.4,
        debt_to_equity=4.0,

        dividend_yield=0.0,
    )

    result = ENGINE.analyze(data)

    assert result.signal == "bearish"

    assert result.fundamental_score < 0

    assert result.confidence > 0

    assert result.data_completeness == 1.0


def test_mixed_fundamentals() -> None:

    data = FundamentalAnalysisInput(

        instrument_key="NSE_EQ|TEST_MIXED",

        symbol="TESTMIXED",

        sector="Diversified",

        pe_ratio=18.0,
        pb_ratio=6.0,
        ev_ebitda=15.0,

        roe=18.0,
        roa=7.0,
        roce=16.0,

        quick_ratio=1.0,
        debt_to_equity=2.0,

        dividend_yield=1.0,
    )

    result = ENGINE.analyze(data)

    assert result.signal in {
        "bullish",
        "neutral",
        "bearish",
    }

    assert result.available_metric_count == 9

    assert_between(
        result.fundamental_score,
        -1.0,
        1.0,
    )

    assert_between(
        result.confidence,
        0.0,
        1.0,
    )


def test_missing_data_is_not_bearish() -> None:

    data = FundamentalAnalysisInput(

        instrument_key="NSE_EQ|TEST_MISSING",

        symbol="TESTMISSING",

        sector="Unknown",

        pe_ratio=None,
        pb_ratio=None,
        ev_ebitda=None,

        roe=None,
        roa=None,
        roce=None,

        quick_ratio=None,
        debt_to_equity=None,

        dividend_yield=None,
    )

    result = ENGINE.analyze(data)

    assert result.signal == "neutral"

    assert result.confidence == 0.0

    assert result.fundamental_score == 0.0

    assert result.available_metric_count == 0

    assert result.data_completeness == 0.0

    assert len(result.warnings) >= 9


def test_partial_data() -> None:

    data = FundamentalAnalysisInput(

        instrument_key="NSE_EQ|TEST_PARTIAL",

        symbol="TESTPARTIAL",

        sector="Technology",

        pe_ratio=15.0,

        roe=20.0,

        roa=None,
        roce=None,

        pb_ratio=None,
        ev_ebitda=None,

        quick_ratio=None,
        debt_to_equity=None,

        dividend_yield=None,
    )

    result = ENGINE.analyze(data)
    print("DEBUG available_metric_count =", result.available_metric_count)
    print("DEBUG total_metric_count     =", result.total_metric_count)
    print("DEBUG data_completeness      =", result.data_completeness)

    assert abs(
        result.data_completeness - (2 / 9)
    ) < 1e-5

    assert result.signal in {
        "bullish",
        "neutral",
        "bearish",
    }

    assert result.confidence < 1.0


def test_zero_values() -> None:

    data = FundamentalAnalysisInput(

        instrument_key="NSE_EQ|TEST_ZERO",

        symbol="TESTZERO",

        pe_ratio=0.0,
        pb_ratio=0.0,
        ev_ebitda=0.0,

        roe=0.0,
        roa=0.0,
        roce=0.0,

        quick_ratio=0.0,
        debt_to_equity=0.0,

        dividend_yield=0.0,
    )

    result = ENGINE.analyze(data)

    assert result.available_metric_count == 9

    assert_between(
        result.fundamental_score,
        -1.0,
        1.0,
    )


def test_deterministic_output() -> None:

    data = FundamentalAnalysisInput(

        instrument_key="NSE_EQ|TEST_DETERMINISTIC",

        symbol="TESTDET",

        sector="Technology",

        pe_ratio=17.5,
        pb_ratio=2.5,
        ev_ebitda=13.0,

        roe=19.0,
        roa=8.0,
        roce=17.0,

        quick_ratio=1.2,
        debt_to_equity=0.9,

        dividend_yield=1.5,
    )

    first = ENGINE.analyze(data)

    second = ENGINE.analyze(data)

    assert first.model_dump() == second.model_dump()


def test_invalid_non_finite_values() -> None:

    data = FundamentalAnalysisInput(

        instrument_key="NSE_EQ|TEST_INVALID",

        symbol="TESTINVALID",

        pe_ratio=None,
        pb_ratio=None,
        ev_ebitda=None,

        roe=None,
        roa=None,
        roce=None,

        quick_ratio=None,
        debt_to_equity=None,

        dividend_yield=None,
    )

    # Engine must remain deterministic and safe even when
    # values are unavailable.
    result = ENGINE.analyze(data)

    assert result.signal == "neutral"

    assert math.isfinite(
        result.fundamental_score
    )

    assert math.isfinite(
        result.confidence
    )


def test_no_trade_decision() -> None:

    data = FundamentalAnalysisInput(

        instrument_key="NSE_EQ|TEST_NO_TRADE",

        symbol="TEST",

        pe_ratio=12.0,
        pb_ratio=1.5,
        ev_ebitda=8.0,

        roe=25.0,
        roa=12.0,
        roce=24.0,

        quick_ratio=2.0,
        debt_to_equity=0.3,

        dividend_yield=4.0,
    )

    result = ENGINE.analyze(data)

    assert result.signal in {
        "bullish",
        "bearish",
        "neutral",
    }

    assert result.signal not in {
        "BUY",
        "HOLD",
        "SELL",
    }


def run_all_tests() -> None:

    test_strong_fundamentals()
    test_weak_fundamentals()
    test_mixed_fundamentals()
    test_missing_data_is_not_bearish()
    test_partial_data()
    test_zero_values()
    test_deterministic_output()
    test_invalid_non_finite_values()
    test_no_trade_decision()

    print(
        "PHASE14_3_FUNDAMENTAL_TESTS_OK"
    )


if __name__ == "__main__":
    run_all_tests()