import unittest

import numpy as np
import pandas as pd

from backend.app.ml.phase15.validation.trading_strategy_simulation import (
    CostConfig,
    calculate_portfolio_metrics,
    fee_breakdown,
    max_affordable_shares,
    simulate_portfolio,
    top_k_fractional_weights,
)


def zero_cost_config():
    return CostConfig(
        broker_name="test", exchange="NSE", brokerage_per_order_inr=0.0,
        stt_buy_bps=0.0, stt_sell_bps=0.0, stamp_duty_buy_bps=0.0,
        exchange_transaction_bps_before_2026_03_01=0.0,
        exchange_transaction_bps_from_2026_03_01=0.0,
        sebi_fee_per_crore_inr=0.0,
        ipft_per_crore_inr=0.0, gst_rate=0.0,
        dp_charge_per_scrip_per_sell_day_inr=0.0,
        slippage_bps_per_side=0.0, max_adv_participation=1.0,
        slippage_scenarios_bps=(0.0,),
    )


class TestPart42Helpers(unittest.TestCase):
    def test_topk_weights_handle_ties_fractionally(self):
        scores = np.array([10, 9, 8, 7, 6, 6, 5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5, -6, -7, -8], dtype=float)
        weights = top_k_fractional_weights(scores, 5)
        self.assertAlmostEqual(weights.sum(), 5.0)
        self.assertAlmostEqual(weights[4], 0.5)
        self.assertAlmostEqual(weights[5], 0.5)

    def test_constant_scores_are_rejected(self):
        with self.assertRaises(ValueError):
            top_k_fractional_weights(np.ones(20), 5)

    def test_fee_breakdown_applies_buy_stamp_and_sell_dp(self):
        cfg = CostConfig(
            broker_name="test", exchange="NSE", brokerage_per_order_inr=20,
            stt_buy_bps=10, stt_sell_bps=10, stamp_duty_buy_bps=1.5,
            exchange_transaction_bps_before_2026_03_01=0.297,
            exchange_transaction_bps_from_2026_03_01=0.307,
            sebi_fee_per_crore_inr=10,
            ipft_per_crore_inr=0.01, gst_rate=0.18,
            dp_charge_per_scrip_per_sell_day_inr=20,
            slippage_bps_per_side=0, max_adv_participation=1,
            slippage_scenarios_bps=(0,),
        )
        buy = fee_breakdown(100_000, "BUY", "2026-02-27", cfg)
        sell = fee_breakdown(100_000, "SELL", "2026-03-02", cfg)
        self.assertGreater(buy["stamp_duty_inr"], 0)
        self.assertEqual(buy["dp_inr"], 0)
        self.assertEqual(sell["stamp_duty_inr"], 0)
        self.assertEqual(sell["dp_inr"], 20)
        before_rate = 100_000 * 0.297 / 10_000
        after_rate = 100_000 * 0.307 / 10_000
        self.assertAlmostEqual(buy["exchange_inr"], before_rate, places=8)
        self.assertAlmostEqual(sell["exchange_inr"], after_rate, places=8)

    def test_share_quantity_respects_budget(self):
        cfg = zero_cost_config()
        qty = max_affordable_shares(1_000, 100, 100, "2026-01-01", cfg)
        self.assertEqual(qty, 10)


class TestPortfolioSimulator(unittest.TestCase):
    def setUp(self):
        self.calendar = pd.bdate_range("2025-01-01", periods=45).strftime("%Y-%m-%d").tolist()
        self.stock_keys = [f"STOCK_{i:02d}" for i in range(20)]
        self.bars = {}
        for i, key in enumerate(self.stock_keys):
            self.bars[key] = {}
            for j, day in enumerate(self.calendar):
                price = 100.0 + i + j * (0.2 + i * 0.001)
                self.bars[key][day] = {
                    "open": price, "high": price + 1, "low": price - 1,
                    "close": price + 0.1, "volume": 100_000.0,
                }
        self.bars["NIFTY50_INDEX"] = {
            day: {"open": 20_000 + j * 5, "high": 20_010 + j * 5,
                  "low": 19_990 + j * 5, "close": 20_005 + j * 5,
                  "volume": 1_000_000.0}
            for j, day in enumerate(self.calendar)
        }
        self.cfg = zero_cost_config()

    def make_signals(self, dates):
        rows = []
        for signal_date in dates:
            for i, key in enumerate(self.stock_keys):
                rows.append({
                    "fold_number": 1, "model": "extra_trees",
                    "instrument_key": key, "session_date": signal_date,
                    "predicted_return": float(20 - i),
                })
        return pd.DataFrame(rows)

    def test_one_trade_exits_after_seven_holding_sessions(self):
        signal_date = self.calendar[22]
        signals = self.make_signals([signal_date])
        equity, trades, skipped = simulate_portfolio(
            "test", "top_k", signals, self.calendar, self.bars,
            self.stock_keys, self.cfg, initial_capital=700_000,
        )
        self.assertEqual(len(trades), 5)
        self.assertTrue((trades["holding_sessions"] == 7).all())
        self.assertTrue((trades["entry_date"] == self.calendar[23]).all())
        self.assertTrue((trades["exit_date"] == self.calendar[29]).all())
        self.assertEqual(int(equity.iloc[-1]["open_positions"]), 0)
        self.assertAlmostEqual(
            equity.iloc[-1]["gross_equity_before_costs_inr"],
            equity.iloc[-1]["net_liquidation_equity_inr"],
            places=6,
        )

    def test_seven_sleeves_prevent_capital_reuse_and_holdings_close(self):
        dates = self.calendar[20:30]
        signals = self.make_signals(dates)
        equity, trades, skipped = simulate_portfolio(
            "test", "top_k", signals, self.calendar, self.bars,
            self.stock_keys, self.cfg, initial_capital=700_000,
        )
        self.assertFalse(equity.empty)
        self.assertLessEqual(int(equity["open_positions"].max()), 35)
        self.assertTrue((trades["holding_sessions"] == 7).all())
        self.assertEqual(int(equity.iloc[-1]["open_positions"]), 0)
        self.assertTrue(skipped.empty)


    def test_net_equity_reconciles_and_costs_reduce_gross_equity(self):
        cfg = CostConfig(
            broker_name="test", exchange="NSE", brokerage_per_order_inr=20.0,
            stt_buy_bps=10.0, stt_sell_bps=10.0, stamp_duty_buy_bps=1.5,
            exchange_transaction_bps_before_2026_03_01=0.297,
            exchange_transaction_bps_from_2026_03_01=0.307,
            sebi_fee_per_crore_inr=10.0, ipft_per_crore_inr=0.0,
            gst_rate=0.18, dp_charge_per_scrip_per_sell_day_inr=20.0,
            slippage_bps_per_side=5.0, max_adv_participation=1.0,
            slippage_scenarios_bps=(5.0,),
        )
        signal_date = self.calendar[22]
        signals = self.make_signals([signal_date])
        equity, trades, _ = simulate_portfolio(
            "costed", "top_k", signals, self.calendar, self.bars,
            self.stock_keys, cfg, initial_capital=700_000,
        )
        metrics = calculate_portfolio_metrics(
            equity, trades, 700_000, "slippage_5bps"
        )
        self.assertAlmostEqual(
            metrics["ending_net_liquidation_equity_inr"] - 700_000,
            metrics["net_trade_pnl_inr"], places=5,
        )
        self.assertAlmostEqual(
            metrics["gross_trade_pnl_inr"] - metrics["net_trade_pnl_inr"],
            metrics["total_transaction_and_slippage_costs_inr"], places=5,
        )
        self.assertGreater(
            metrics["ending_gross_equity_inr"],
            metrics["ending_net_liquidation_equity_inr"],
        )


    def test_constant_signal_date_is_skipped_not_fatal(self):
        signal_date = self.calendar[22]
        signals = self.make_signals([signal_date])
        signals["predicted_return"] = 0.01
        equity, trades, skipped = simulate_portfolio(
            "constant", "top_k", signals, self.calendar, self.bars,
            self.stock_keys, self.cfg, initial_capital=700_000,
        )
        self.assertTrue(trades.empty)
        self.assertEqual(len(skipped), 1)
        self.assertEqual(
            skipped.iloc[0]["reason"],
            "constant_predictions_no_cross_sectional_rank",
        )
        self.assertAlmostEqual(
            equity.iloc[-1]["net_liquidation_equity_inr"],
            700_000.0, places=6,
        )

    def test_equal_weight_benchmark_buys_all_20(self):
        signals = self.make_signals([self.calendar[22]])[["fold_number", "session_date"]].drop_duplicates()
        equity, trades, _ = simulate_portfolio(
            "equal", "equal_weight_20", signals, self.calendar,
            self.bars, self.stock_keys, self.cfg, initial_capital=700_000,
        )
        self.assertEqual(len(trades), 20)
        self.assertEqual(int(equity.iloc[-1]["open_positions"]), 0)


if __name__ == "__main__":
    unittest.main()
