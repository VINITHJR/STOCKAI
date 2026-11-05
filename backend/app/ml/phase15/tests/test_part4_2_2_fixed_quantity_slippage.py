import unittest

import numpy as np
import pandas as pd

from backend.app.ml.phase15.validation import part4_2_1_diagnostics as p421
from backend.app.ml.phase15.validation.part4_2_2_fixed_quantity_slippage import (
    ANCHOR_SLIPPAGE_BPS,
    PORTFOLIOS,
    SCENARIOS_BPS,
    compare_anchor_replay,
    ensure_daily_equity_scenario_labels,
    make_fixed_quantity_schedule,
    replay_fixed_schedule,
    schedule_signature,
)


def make_fee_schedule():
    return p421.FeeSchedule(
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


def synthetic_market():
    calendar = pd.bdate_range("2025-01-01", periods=60).strftime("%Y-%m-%d").tolist()
    stock_keys = [f"STOCK_{i:02d}" for i in range(20)]
    bars = {}
    for i, key in enumerate(stock_keys):
        bars[key] = {}
        base = 100.0 + i * 2.0
        for j, day in enumerate(calendar):
            open_price = base + j * (0.15 + i * 0.001)
            close_price = open_price + (0.20 if i % 2 == 0 else -0.05) + j * 0.002
            bars[key][day] = {
                "open": open_price,
                "high": max(open_price, close_price) + 1.0,
                "low": min(open_price, close_price) - 1.0,
                "close": close_price,
                "volume": 100_000.0,
            }
    return calendar, stock_keys, bars


def make_anchor_from_original_simulator():
    calendar, stock_keys, bars = synthetic_market()
    fee_schedule = make_fee_schedule()
    signal_date = calendar[25]
    model_signals = pd.DataFrame([
        {
            "fold_number": 1,
            "model": model,
            "instrument_key": key,
            "session_date": signal_date,
            "predicted_return": float(30 - i) + (0.1 if model == "random_forest" else 0.0),
        }
        for model in ("extra_trees", "random_forest")
        for i, key in enumerate(stock_keys)
    ])
    benchmark_signals = pd.DataFrame([{
        "fold_number": 1,
        "session_date": signal_date,
    }])

    source_trades = []
    source_equity = []
    source_metrics = []
    signal_map = {
        "extra_trees_top5": ("top_k", model_signals[model_signals.model == "extra_trees"].copy()),
        "random_forest_top5": ("top_k", model_signals[model_signals.model == "random_forest"].copy()),
        "equal_weight_20_integer_redistributed": ("equal_weight_redistributed", benchmark_signals),
    }
    for portfolio, (kind, signals) in signal_map.items():
        equity, trades, _, _, _ = p421.simulate_variant(
            portfolio, kind, signals, calendar, bars, stock_keys, fee_schedule,
            ANCHOR_SLIPPAGE_BPS,
        )
        source_trades.append(trades)
        equity = equity.copy()
        equity["slippage_bps_per_side"] = ANCHOR_SLIPPAGE_BPS
        source_equity.append(equity)
        source_metrics.append(p421._equity_metrics(
            equity, trades, p421.INITIAL_CAPITAL_INR, portfolio, str(ANCHOR_SLIPPAGE_BPS)
        ))

    source_trades_df = pd.concat(source_trades, ignore_index=True)
    fixed_schedule = make_fixed_quantity_schedule(
        source_trades_df, calendar, bars, fee_schedule
    )
    return calendar, stock_keys, bars, fee_schedule, source_trades_df, pd.concat(source_equity, ignore_index=True), pd.DataFrame(source_metrics), fixed_schedule


class TestPart422Schedule(unittest.TestCase):
    def setUp(self):
        (self.calendar, self.stock_keys, self.bars, self.fees,
         self.source_trades, self.source_equity, self.source_metrics,
         self.fixed_schedule) = make_anchor_from_original_simulator()

    def test_20bps_anchor_reproduces_original_part421_trades_quantities_and_metrics(self):
        equity_frames = []
        trade_frames = []
        metric_rows = []
        for portfolio in PORTFOLIOS:
            equity, trades, _ = replay_fixed_schedule(
                portfolio, self.fixed_schedule, self.calendar, self.bars,
                self.fees, ANCHOR_SLIPPAGE_BPS,
            )
            equity_frames.append(equity)
            trade_frames.append(trades)
            metric_rows.append(p421._equity_metrics(
                equity, trades, p421.INITIAL_CAPITAL_INR, portfolio,
                str(ANCHOR_SLIPPAGE_BPS),
            ))
        result = compare_anchor_replay(
            pd.DataFrame(metric_rows), self.source_metrics,
            pd.concat(equity_frames, ignore_index=True), self.source_equity,
            pd.concat(trade_frames, ignore_index=True), self.source_trades,
        )
        self.assertEqual(len(result), 3)
        self.assertTrue(result["all_metric_checks_pass"].all())

    def test_same_selections_quantities_sleeves_and_gross_pnl_for_all_scenarios(self):
        for portfolio in PORTFOLIOS:
            outputs = []
            for slip in SCENARIOS_BPS:
                _, trades, _ = replay_fixed_schedule(
                    portfolio, self.fixed_schedule, self.calendar, self.bars,
                    self.fees, slip,
                )
                outputs.append(trades)
            hashes = [schedule_signature(x) for x in outputs]
            self.assertEqual(len(set(hashes)), 1)
            gross = [float(x["gross_pnl_inr"].sum()) for x in outputs]
            self.assertTrue(np.allclose(gross, gross[0], rtol=1e-10, atol=0.01))

    def test_net_pnl_deteriorates_as_slippage_increases(self):
        for portfolio in PORTFOLIOS:
            net = []
            for slip in SCENARIOS_BPS:
                _, trades, _ = replay_fixed_schedule(
                    portfolio, self.fixed_schedule, self.calendar, self.bars,
                    self.fees, slip,
                )
                net.append(float(trades["net_pnl_inr"].sum()))
            self.assertTrue(np.all(np.diff(net) <= 0.05), msg=f"{portfolio}: {net}")

    def test_zero_slippage_keeps_brokerage_and_statutory_fees(self):
        equity, trades, orders = replay_fixed_schedule(
            PORTFOLIOS[0], self.fixed_schedule, self.calendar, self.bars,
            self.fees, 0.0,
        )
        self.assertGreater(float(trades["total_fees_inr"].sum()), 0.0)
        self.assertAlmostEqual(float(trades["total_slippage_inr"].sum()), 0.0, places=10)
        self.assertEqual(set(orders["side"].unique()), {"BUY", "SELL"})
        self.assertEqual(int(equity.iloc[-1]["open_stock_legs"]), 0)

    def test_same_number_of_trade_legs_for_all_slippage_scenarios(self):
        for portfolio in PORTFOLIOS:
            counts = []
            for slip in SCENARIOS_BPS:
                _, trades, _ = replay_fixed_schedule(
                    portfolio, self.fixed_schedule, self.calendar, self.bars,
                    self.fees, slip,
                )
                counts.append(len(trades))
            self.assertEqual(len(set(counts)), 1)

    def test_invalid_scenario_is_rejected(self):
        with self.assertRaises(ValueError):
            replay_fixed_schedule(
                PORTFOLIOS[0], self.fixed_schedule, self.calendar,
                self.bars, self.fees, 7.5,
            )


class TestDailyEquityScenarioRecovery(unittest.TestCase):
    def test_missing_scenario_labels_are_recovered_only_from_verified_block_order(self):
        dates = ["2025-01-01", "2025-01-02", "2025-01-03"]
        portfolios = []
        for scenario in SCENARIOS_BPS:
            for portfolio in PORTFOLIOS:
                for day in dates:
                    portfolios.append({
                        "portfolio": portfolio, "session_date": day,
                        "net_liquidation_equity_inr": 1_000_000.0,
                        "gross_equity_before_costs_inr": 1_000_000.0,
                    })
        daily = pd.DataFrame(portfolios)
        metrics = pd.DataFrame([
            {"portfolio": portfolio, "slippage_bps_per_side": scenario}
            for portfolio in PORTFOLIOS for scenario in SCENARIOS_BPS
        ])
        result = ensure_daily_equity_scenario_labels(daily, metrics)
        self.assertEqual(set(result["slippage_bps_per_side"].unique()), set(SCENARIOS_BPS))
        for portfolio in PORTFOLIOS:
            self.assertEqual(
                result[result.portfolio == portfolio].groupby("slippage_bps_per_side").size().to_dict(),
                {float(s): 3 for s in SCENARIOS_BPS},
            )

    def test_daily_equity_scenario_recovery_rejects_misaligned_blocks(self):
        dates = ["2025-01-01", "2025-01-02", "2025-01-03"]
        rows = []
        for scenario_idx, scenario in enumerate(SCENARIOS_BPS):
            for portfolio in PORTFOLIOS:
                block_dates = dates.copy()
                if scenario_idx == 2 and portfolio == PORTFOLIOS[0]:
                    block_dates[1] = "2025-01-04"
                for day in block_dates:
                    rows.append({
                        "portfolio": portfolio, "session_date": day,
                        "net_liquidation_equity_inr": 1_000_000.0,
                        "gross_equity_before_costs_inr": 1_000_000.0,
                    })
        daily = pd.DataFrame(rows)
        metrics = pd.DataFrame([
            {"portfolio": portfolio, "slippage_bps_per_side": scenario}
            for portfolio in PORTFOLIOS for scenario in SCENARIOS_BPS
        ])
        with self.assertRaises(ValueError):
            ensure_daily_equity_scenario_labels(daily, metrics)


if __name__ == "__main__":
    unittest.main(verbosity=2)
