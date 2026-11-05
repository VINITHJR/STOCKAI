import unittest

import numpy as np
import pandas as pd

from backend.app.ml.phase15.validation import part4_3_risk_stability as p43


class TestDrawdownDiagnostics(unittest.TestCase):
    def test_drawdown_episodes_record_recovery_and_peak_to_recovery(self):
        dates = pd.bdate_range("2025-01-01", periods=7).strftime("%Y-%m-%d").tolist()
        equity = np.array([100.0, 90.0, 95.0, 100.0, 105.0, 100.0, 110.0])
        episodes, hwm, dd = p43.drawdown_episodes(dates, equity, initial_capital=100.0)
        self.assertEqual(len(episodes), 2)
        self.assertTrue(episodes["recovered"].all())
        self.assertEqual(episodes.iloc[0]["recovery_date"], dates[3])
        self.assertEqual(int(episodes.iloc[0]["underwater_sessions"]), 2)
        self.assertEqual(episodes.iloc[1]["recovery_date"], dates[6])
        self.assertAlmostEqual(float(dd.min()), -0.10)
        self.assertEqual(float(hwm[-1]), 110.0)

    def test_unrecovered_drawdown_is_explicit(self):
        dates = pd.bdate_range("2025-02-03", periods=4).strftime("%Y-%m-%d").tolist()
        equity = np.array([100.0, 95.0, 90.0, 92.0])
        episodes, _, _ = p43.drawdown_episodes(dates, equity, initial_capital=100.0)
        self.assertEqual(len(episodes), 1)
        self.assertFalse(bool(episodes.iloc[0]["recovered"]))
        self.assertEqual(episodes.iloc[0]["status"], "UNRECOVERED_AT_END")
        self.assertIsNone(episodes.iloc[0]["recovery_date"])
        self.assertEqual(int(episodes.iloc[0]["underwater_sessions"]), 3)

    def test_daily_risk_metrics_return_and_cvar(self):
        dates = pd.bdate_range("2025-03-03", periods=6).strftime("%Y-%m-%d")
        equity_values = [1_000_000, 1_010_000, 990_000, 995_000, 980_000, 1_005_000]
        frame = pd.DataFrame({
            "portfolio": "extra_trees_top5",
            "slippage_bps_per_side": 0.0,
            "session_date": dates,
            "cash_inr": 100_000.0,
            "open_stock_legs": 5,
            "raw_market_value_inr": 900_000.0,
            "gross_equity_before_costs_inr": equity_values,
            "net_liquidation_equity_inr": equity_values,
            "daily_exposure_pct": 90.0,
        })
        metrics, daily, episodes = p43.calculate_daily_risk_metrics(frame)
        self.assertAlmostEqual(metrics["net_total_return_pct"], 0.5, places=8)
        self.assertAlmostEqual(metrics["max_drawdown_pct"], (980000 / 1010000 - 1) * 100, places=8)
        self.assertEqual(metrics["daily_return_observations"], 5)
        self.assertGreater(metrics["daily_cvar_95_loss_pct"], 0)
        self.assertEqual(len(daily), 6)
        self.assertGreaterEqual(len(episodes), 1)


class TestStockAndFoldDiagnostics(unittest.TestCase):
    def make_trades(self):
        return pd.DataFrame([
            {"portfolio": "extra_trees_top5", "slippage_bps_per_side": 0.0, "fold_number": 1,
             "instrument_key": "A", "signal_date": "2025-01-01", "net_pnl_inr": -100.0,
             "gross_pnl_inr": 20.0, "total_fees_inr": 80.0, "total_slippage_inr": 40.0,
             "net_return_on_entry_cash_pct": -1.0},
            {"portfolio": "extra_trees_top5", "slippage_bps_per_side": 0.0, "fold_number": 1,
             "instrument_key": "A", "signal_date": "2025-01-02", "net_pnl_inr": -50.0,
             "gross_pnl_inr": 30.0, "total_fees_inr": 80.0, "total_slippage_inr": 0.0,
             "net_return_on_entry_cash_pct": -0.5},
            {"portfolio": "extra_trees_top5", "slippage_bps_per_side": 0.0, "fold_number": 1,
             "instrument_key": "B", "signal_date": "2025-01-03", "net_pnl_inr": 150.0,
             "gross_pnl_inr": 200.0, "total_fees_inr": 50.0, "total_slippage_inr": 0.0,
             "net_return_on_entry_cash_pct": 1.5},
        ])

    def test_stock_loss_concentration_hhi(self):
        result = p43.build_stock_concentration(self.make_trades())
        a = result[result.instrument_key == "A"].iloc[0]
        b = result[result.instrument_key == "B"].iloc[0]
        self.assertEqual(int(a["completed_stock_legs"]), 2)
        self.assertAlmostEqual(a["share_of_all_negative_trade_loss_pct"], 100.0)
        self.assertAlmostEqual(b["share_of_all_negative_trade_loss_pct"], 0.0)
        self.assertAlmostEqual(a["portfolio_stock_loss_hhi"], 1.0)
        self.assertAlmostEqual(a["effective_number_of_loss_contributors"], 1.0)

    def test_fold_cohort_trade_attribution_reconciles(self):
        trades = self.make_trades()
        regimes = pd.DataFrame({"session_date": ["2025-01-01"], "fold_number": [1]})
        daily = pd.DataFrame({
            "portfolio": ["extra_trees_top5", "extra_trees_top5"],
            "slippage_bps_per_side": [0.0, 0.0],
            "session_date": ["2025-01-01", "2025-01-02"],
            "daily_return_pct": [0.0, -0.1],
            "daily_exposure_pct": [0.0, 80.0],
        })
        cohort, calendar = p43.build_fold_reports(trades, daily, regimes)
        self.assertEqual(len(cohort), 1)
        self.assertAlmostEqual(cohort.iloc[0]["net_pnl_inr"], 0.0)
        self.assertEqual(cohort.iloc[0]["completed_stock_legs"], 3)
        self.assertEqual(len(calendar), 1)
        self.assertEqual(calendar.iloc[0]["fold_type"], "calendar_period_mark_to_market_proxy")


class TestMarketAndReliabilityDiagnostics(unittest.TestCase):
    def test_reliability_state_uses_only_matured_past_observations(self):
        dates = pd.bdate_range("2024-01-01", periods=303).strftime("%Y-%m-%d")
        rows = []
        for model in ("extra_trees", "random_forest"):
            for i, day in enumerate(dates):
                value = -0.02 if i < 40 else 0.04
                rows.append({
                    "experiment": "primary", "model": model,
                    "fold_number": 1 if i < 61 else 2,
                    "session_date": day,
                    "cross_sectional_spearman_ic": value,
                    "top_minus_bottom_actual_return": value / 10,
                })
        reliability = p43.build_prediction_reliability(pd.DataFrame(rows))
        et = reliability[reliability.model == "extra_trees"].sort_values("session_date")
        self.assertEqual(et.iloc[0]["prediction_reliability_state"], "insufficient_history")
        # Row 26 has 20 matured observations from rows 0..19 (seven-session maturity lag).
        self.assertEqual(et.iloc[26]["prediction_reliability_state"], "weak_trailing_reliability")
        self.assertLess(et.iloc[26]["matured_through_signal_date"], et.iloc[26]["session_date"])
        self.assertEqual(et.iloc[100]["prediction_reliability_state"], "positive_trailing_reliability")

    def test_fold_specific_vix_thresholds_use_training_history_only(self):
        market_dates = pd.date_range("2023-10-01", "2026-04-16", freq="D").strftime("%Y-%m-%d")
        market = pd.DataFrame({
            "session_date": market_dates,
            "india_vix_close": [10.0 + (i % 31) * 0.25 for i in range(len(market_dates))],
        })
        rows = []
        elevated_count = 0
        market_index = market.set_index("session_date")["india_vix_close"]
        for fold, (start, end, expected_count) in p43.EXPECTED_FOLD_DATES.items():
            dates = pd.date_range(start, end, periods=expected_count).strftime("%Y-%m-%d")
            # Match Part 3.5: use only the 120 sessions strictly
            # before the validation start date.
            training = market[market["session_date"] < start].tail(120)
            self.assertEqual(len(training), 120)

            threshold = float(training["india_vix_close"].quantile(0.75))
            elevated_count += int(
                (market_index.loc[dates].to_numpy() > threshold).sum()
            )
            for model in ("extra_trees", "random_forest"):
                for i, day in enumerate(dates):
                    rows.append({
                        "experiment": "primary", "fold_number": fold, "model": model,
                        "session_date": day,
                        "nifty_trend_state": ("bullish", "bearish", "mixed")[i % 3],
                        "market_risk_state": ("risk_on", "risk_off", "mixed")[i % 3],
                        "cross_sectional_spearman_ic": 0.02,
                        "top_minus_bottom_actual_return": 0.001,
                    })
        sample_sizes = pd.DataFrame([{
            "regime_type": "vix_level_vs_training_q75",
            "regime": "elevated",
            "validation_dates": elevated_count,
        }])
        regimes, thresholds = p43.build_market_regimes(pd.DataFrame(rows), market, sample_sizes)
        self.assertEqual(len(regimes), 303)
        self.assertEqual(len(thresholds), 5)
        self.assertEqual(int((regimes["vix_level_regime"] == "elevated").sum()), elevated_count)
        self.assertTrue((thresholds["training_observations"] == 120).all())

    def test_regime_trade_report_attributes_whole_trade_to_signal_date(self):
        trades = pd.DataFrame([{
            "portfolio": "extra_trees_top5", "slippage_bps_per_side": 0.0,
            "fold_number": 1, "signal_date": "2025-01-14", "instrument_key": "A",
            "gross_pnl_inr": 100.0, "net_pnl_inr": 50.0, "total_fees_inr": 50.0,
            "total_slippage_inr": 0.0, "net_return_on_entry_cash_pct": 0.5,
        }])
        regimes = pd.DataFrame([{
            "session_date": "2025-01-14", "fold_number": 1,
            "nifty_trend_state": "bullish", "market_risk_state": "risk_on",
            "vix_level_regime": "normal", "india_vix_close": 14.0,
            "vix_training_q75": 18.0,
        }])
        report = p43.build_regime_trade_report(trades, regimes)
        self.assertEqual(len(report), 3)
        nifty = report[report.regime_type == "nifty_trend"].iloc[0]
        self.assertEqual(nifty["regime"], "bullish")
        self.assertAlmostEqual(nifty["net_pnl_inr"], 50.0)

    def test_market_regime_exposure_reports_daily_mark_to_market_exposure(self):
        daily = pd.DataFrame([
            {"portfolio": "extra_trees_top5", "slippage_bps_per_side": 0.0,
             "session_date": "2025-01-01", "daily_return_pct": -0.2, "daily_exposure_pct": 80.0},
            {"portfolio": "extra_trees_top5", "slippage_bps_per_side": 0.0,
             "session_date": "2025-01-02", "daily_return_pct": 0.1, "daily_exposure_pct": 20.0},
        ])
        regimes = pd.DataFrame([
            {"session_date": "2025-01-01", "nifty_trend_state": "bearish",
             "market_risk_state": "risk_off", "vix_level_regime": "elevated"},
            {"session_date": "2025-01-02", "nifty_trend_state": "bullish",
             "market_risk_state": "risk_on", "vix_level_regime": "normal"},
        ])
        result = p43.build_regime_exposure_report(daily, regimes)
        self.assertEqual(len(result), 6)
        elevated = result[(result.regime_type == "vix_level_vs_training_q75") & (result.regime == "elevated")].iloc[0]
        self.assertAlmostEqual(elevated["average_exposure_pct"], 80.0)
        self.assertAlmostEqual(elevated["mean_daily_return_pct"], -0.2)

    def test_reliability_exposure_comparison_is_scenario_specific(self):
        daily_rows = []
        for slip in (0.0, 5.0):
            for i, day in enumerate(("2025-01-01", "2025-01-02", "2025-01-03", "2025-01-06")):
                daily_rows.append({
                    "portfolio": "extra_trees_top5", "slippage_bps_per_side": slip,
                    "session_date": day, "daily_return_pct": 0.1,
                    "daily_exposure_pct": [80.0, 80.0, 20.0, 20.0][i] + slip,
                })
        daily = pd.DataFrame(daily_rows)
        reliability = pd.DataFrame({
            "model": ["extra_trees"] * 4,
            "session_date": ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-06"],
            "prediction_reliability_state": [
                "weak_trailing_reliability", "weak_trailing_reliability",
                "positive_trailing_reliability", "positive_trailing_reliability",
            ],
            "trailing_matured_mean_ic_20": [-0.1, -0.1, 0.1, 0.1],
            "trailing_matured_mean_top_bottom_spread_20": [-0.01, -0.01, 0.01, 0.01],
            "matured_through_signal_date": ["2024-12-20"] * 4,
        })
        exposure, comparison = p43.build_reliability_exposure_report(daily, reliability)
        self.assertEqual(len(comparison), 2)
        for row in comparison.itertuples(index=False):
            self.assertAlmostEqual(row.weak_minus_positive_exposure_pp, 60.0)

    def test_metric_reconciliation_passes_matching_source_values(self):
        calculated = pd.DataFrame([{
            "portfolio": p, "slippage_bps_per_side": s,
            "net_total_return_pct": -10.0, "gross_total_return_before_costs_pct": 5.0,
            "max_drawdown_pct": -20.0, "annualized_volatility_pct": 15.0,
            "sharpe_rf_zero": -0.5, "average_daily_exposure_pct": 60.0,
        } for p in p43.PORTFOLIOS for s in p43.SCENARIOS_BPS])
        source = calculated.copy()
        result = p43.reconcile_existing_portfolio_metrics(calculated, source)
        self.assertEqual(len(result), 12)
        self.assertTrue(result["passed"].all())


if __name__ == "__main__":
    unittest.main()
