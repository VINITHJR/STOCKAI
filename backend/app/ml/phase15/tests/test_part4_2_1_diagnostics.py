import unittest

import numpy as np
import pandas as pd

from backend.app.ml.phase15.validation.part4_2_1_diagnostics import (
    FeeSchedule,
    compare_original_equal_weight,
    fee_breakdown_21,
    fold_attribution,
    plan_redistributed_equal_weight,
    simulate_variant,
)


def test_schedule():
    return FeeSchedule(
        brokerage_per_order_inr=20.0,
        stt_buy_bps=10.0,
        stt_sell_bps=10.0,
        stamp_duty_buy_bps=1.5,
        exchange_before_per_crore_inr=297.0,
        ipft_before_per_crore_inr=10.0,
        exchange_from_per_crore_inr=306.99,
        ipft_from_per_crore_inr=0.01,
        sebi_per_crore_inr=10.0,
        dp_per_scrip_sell_day_inr=20.0,
        gst_rate=0.18,
        max_adv_participation=0.01,
        effective_change_date="2026-03-01",
    )


class TestPart421Fees(unittest.TestCase):
    def test_pre_march_2026_exchange_and_ipft_split(self):
        fee = fee_breakdown_21(10_000_000, "BUY", "2026-02-28", test_schedule())
        self.assertAlmostEqual(fee["exchange_transaction_inr"], 297.0)
        self.assertAlmostEqual(fee["ipft_inr"], 10.0)
        self.assertAlmostEqual(fee["stt_inr"], 10_000.0)
        self.assertAlmostEqual(fee["stamp_duty_inr"], 1_500.0)
        self.assertEqual(fee["dp_charge_inr"], 0.0)
        self.assertAlmostEqual(fee["gst_inr"], (20 + 297 + 10) * 0.18)

    def test_post_march_2026_exchange_and_ipft_split(self):
        fee = fee_breakdown_21(10_000_000, "SELL", "2026-03-02", test_schedule())
        self.assertAlmostEqual(fee["exchange_transaction_inr"], 306.99)
        self.assertAlmostEqual(fee["ipft_inr"], 0.01)
        self.assertEqual(fee["stamp_duty_inr"], 0.0)
        self.assertEqual(fee["dp_charge_inr"], 20.0)
        self.assertAlmostEqual(fee["gst_inr"], (20 + 306.99 + 0.01 + 20) * 0.18)
        self.assertAlmostEqual(
            fee["total_fees_inr"],
            sum(fee[k] for k in (
                "brokerage_inr", "exchange_transaction_inr", "ipft_inr",
                "sebi_fee_inr", "stt_inr", "stamp_duty_inr", "dp_charge_inr", "gst_inr"
            )),
        )

    def test_negative_notional_is_rejected(self):
        with self.assertRaises(ValueError):
            fee_breakdown_21(-1, "BUY", "2026-01-01", test_schedule())


class TestRedistributedBenchmark(unittest.TestCase):
    def test_unaffordable_stock_is_removed_and_cash_redistributed(self):
        schedule = test_schedule()
        candidates = [
            {"instrument_key": "A", "fill_price": 10.0, "volume_cap": 1000},
            {"instrument_key": "B", "fill_price": 20.0, "volume_cap": 1000},
            {"instrument_key": "C", "fill_price": 1000.0, "volume_cap": 1000},
        ]
        active, excluded, per_name = plan_redistributed_equal_weight(
            candidates, 300.0, "2026-01-05", schedule
        )
        self.assertEqual({x["instrument_key"] for x in active}, {"A", "B"})
        self.assertEqual(len(excluded), 1)
        self.assertEqual(excluded[0]["instrument_key"], "C")
        self.assertAlmostEqual(per_name, 150.0)
        self.assertAlmostEqual(sum(x["allocation_budget_inr"] for x in active), 300.0)

    def test_adv_ineligible_stock_is_removed(self):
        candidates = [
            {"instrument_key": "A", "fill_price": 10.0, "volume_cap": 0},
            {"instrument_key": "B", "fill_price": 10.0, "volume_cap": 100},
        ]
        active, excluded, per_name = plan_redistributed_equal_weight(
            candidates, 200.0, "2026-01-05", test_schedule()
        )
        self.assertEqual([x["instrument_key"] for x in active], ["B"])
        self.assertEqual(excluded[0]["reason"], "adv20_cap_below_one_share")
        self.assertAlmostEqual(per_name, 200.0)


class TestPart421Simulation(unittest.TestCase):
    def setUp(self):
        self.calendar = pd.bdate_range("2025-01-01", periods=50).strftime("%Y-%m-%d").tolist()
        self.stock_keys = [f"STOCK_{i:02d}" for i in range(20)]
        self.bars = {}
        for i, key in enumerate(self.stock_keys):
            self.bars[key] = {}
            price = 100.0 + i
            for j, day in enumerate(self.calendar):
                p = price + j * 0.1
                self.bars[key][day] = {
                    "open": p, "high": p + 1, "low": p - 1,
                    "close": p + 0.25, "volume": 100_000.0,
                }
        # One equity costs more than a typical 1/20 sleeve slot, causing a
        # redistribution to the other 19 names in the equal-weight reference.
        expensive = self.stock_keys[-1]
        for day in self.calendar:
            self.bars[expensive][day] = {
                "open": 15_000.0, "high": 15_100.0, "low": 14_900.0,
                "close": 15_020.0, "volume": 100_000.0,
            }
        self.schedule = test_schedule()

    def make_signals(self, signal_date):
        return pd.DataFrame([
            {"fold_number": 1, "model": "extra_trees", "instrument_key": key,
             "session_date": signal_date, "predicted_return": float(20 - i)}
            for i, key in enumerate(self.stock_keys)
        ])

    def test_zero_slippage_retains_brokerage_and_statutory_fees(self):
        signal_date = self.calendar[21]
        eq, trades, skipped, orders, audit = simulate_variant(
            "extra_trees_top5", "top_k", self.make_signals(signal_date),
            self.calendar, self.bars, self.stock_keys, self.schedule, 0.0,
        )
        self.assertEqual(len(trades), 5)
        self.assertGreater(trades["total_fees_inr"].sum(), 0)
        self.assertAlmostEqual(trades["total_slippage_inr"].sum(), 0.0)
        self.assertTrue((trades["holding_sessions"] == 7).all())
        self.assertEqual(eq.iloc[-1]["open_stock_legs"], 0)
        self.assertEqual(set(orders["side"].unique()), {"BUY", "SELL"})

    def test_integer_equal_weight_redistributes_and_records_audit(self):
        signal_date = self.calendar[21]
        signals = self.make_signals(signal_date)[["fold_number", "session_date"]].drop_duplicates()
        eq, trades, skipped, orders, audit = simulate_variant(
            "equal_weight_20_integer_redistributed", "equal_weight_redistributed",
            signals, self.calendar, self.bars, self.stock_keys, self.schedule, 0.0,
        )
        self.assertEqual(len(trades), 19)
        self.assertEqual(len(audit), 1)
        self.assertEqual(int(audit.iloc[0]["requested_stock_count"]), 20)
        self.assertEqual(int(audit.iloc[0]["eligible_stock_count_after_redistribution"]), 19)
        self.assertEqual(int(audit.iloc[0]["filled_stock_count"]), 19)
        self.assertEqual(len(skipped), 1)
        self.assertEqual(skipped.iloc[0]["reason"], "one_share_exceeds_equal_weight_budget")
        self.assertEqual(eq.iloc[-1]["open_stock_legs"], 0)

    def test_constant_model_score_date_is_skipped_not_fatal(self):
        signal_date = self.calendar[21]
        signals = self.make_signals(signal_date)
        signals["predicted_return"] = 0.01
        eq, trades, skipped, orders, audit = simulate_variant(
            "extra_trees_top5", "top_k", signals, self.calendar,
            self.bars, self.stock_keys, self.schedule, 0.0,
        )
        self.assertTrue(trades.empty)
        self.assertEqual(skipped.iloc[0]["reason"], "constant_predictions_no_cross_sectional_rank")
        self.assertEqual(eq.iloc[-1]["open_stock_legs"], 0)

    def test_fold_attribution_reconciles_pnl(self):
        signal_date = self.calendar[21]
        eq, trades, _, _, _ = simulate_variant(
            "extra_trees_top5", "top_k", self.make_signals(signal_date),
            self.calendar, self.bars, self.stock_keys, self.schedule, 5.0,
        )
        fold = fold_attribution(trades, 1_000_000.0)
        self.assertEqual(len(fold), 1)
        self.assertAlmostEqual(fold.iloc[0]["gross_pnl_inr"], trades["gross_pnl_inr"].sum())
        self.assertAlmostEqual(fold.iloc[0]["net_pnl_inr"], trades["net_pnl_inr"].sum())


class TestReportComparisons(unittest.TestCase):
    def test_original_benchmark_scenario_labels_match_new_numeric_scenarios(self):
        old = pd.DataFrame([{
            "portfolio": "equal_weight_20_stock",
            "slippage_scenario": "slippage_5bps_per_side",
            "net_total_return_pct": -30.0,
            "gross_total_return_before_costs_pct": 5.0,
            "average_daily_exposure_pct": 50.0,
            "completed_stock_legs": 100,
        }])
        new = pd.DataFrame([{
            "portfolio": "equal_weight_20_integer_redistributed",
            "slippage_bps_per_side": 5.0,
            "net_total_return_pct": -10.0,
            "gross_total_return_before_costs_pct": 6.0,
            "average_daily_exposure_pct": 70.0,
            "completed_stock_legs": 120,
        }])
        result = compare_original_equal_weight(old, new)
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result.iloc[0]["net_return_change_pp"], 20.0)
        self.assertEqual(result.iloc[0]["completed_legs_change"], 20)


if __name__ == "__main__":
    unittest.main(verbosity=2)
