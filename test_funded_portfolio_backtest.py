import unittest

import numpy as np
import pandas as pd

from funded_portfolio_backtest import select_score_thresholds, simulate


class FundedPortfolioBacktestTests(unittest.TestCase):
    def test_robust_threshold_selection_uses_daily_score_scale(self):
        group = pd.DataFrame(
            {
                "Ticker": [f"T{i}" for i in range(7)],
                "prediction": [-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0],
            }
        )
        long, short, median, scale = select_score_thresholds(
            group, "prediction", long_z=1.5, short_z=1.5
        )
        self.assertEqual(long["Ticker"].tolist(), ["T6"])
        self.assertEqual(short["Ticker"].tolist(), ["T0"])
        self.assertEqual(median, 0.0)
        self.assertGreater(scale, 0.0)

    def test_long_only_compounds_daily_mark_to_market_returns(self):
        dates = pd.date_range("2025-01-02", periods=3, freq="B")
        tickers = [f"T{i:02d}" for i in range(20)]
        prediction_rows = []
        price_rows = []
        for day_index, day in enumerate(dates):
            for ticker_index, ticker in enumerate(tickers):
                prediction_rows.append(
                    {
                        "Date": day,
                        "target_end_date": day,
                        "Ticker": ticker,
                        "prediction": ticker_index,
                    }
                )
                gain = 0.10 if ticker_index >= 18 else 0.0
                price_rows.append(
                    {
                        "Date": day,
                        "Ticker": ticker,
                        "Open": 100.0,
                        "Close": 100.0 * (1.0 + gain),
                    }
                )
        result = simulate(
            pd.DataFrame(prediction_rows),
            pd.DataFrame(price_rows),
            mode="long-only",
            horizon=1,
            tail_fraction=0.10,
            cost_bps=0.0,
        )
        self.assertTrue(np.allclose(result["daily_return"], 0.10))
        self.assertAlmostEqual(result["nav"].iloc[-1], 1.1 ** 3)

    def test_long_short_splits_gross_sleeve_equally(self):
        day = pd.Timestamp("2025-01-02")
        predictions = pd.DataFrame(
            {
                "Date": [day] * 20,
                "target_end_date": [day] * 20,
                "Ticker": [f"T{i:02d}" for i in range(20)],
                "prediction": np.arange(20),
            }
        )
        prices = pd.DataFrame(
            {
                "Date": [day] * 20,
                "Ticker": [f"T{i:02d}" for i in range(20)],
                "Open": [100.0] * 20,
                "Close": [
                    90.0 if i < 2 else 110.0 if i >= 18 else 100.0
                    for i in range(20)
                ],
            }
        )
        result = simulate(
            predictions,
            prices,
            mode="long-short",
            horizon=1,
            tail_fraction=0.10,
            cost_bps=0.0,
        )
        # Half of NAV gains 10% long and half gains 10% short.
        self.assertAlmostEqual(result["daily_return"].iloc[0], 0.10)
        self.assertAlmostEqual(result["net_exposure"].iloc[0], 0.0, places=12)

    def test_asymmetric_cutoffs_change_names_not_side_notional(self):
        day = pd.Timestamp("2025-01-02")
        predictions = pd.DataFrame(
            {
                "Date": [day] * 20,
                "target_end_date": [day] * 20,
                "Ticker": [f"T{i:02d}" for i in range(20)],
                "prediction": np.arange(20),
            }
        )
        prices = pd.DataFrame(
            {
                "Date": [day] * 20,
                "Ticker": [f"T{i:02d}" for i in range(20)],
                "Open": [100.0] * 20,
                "Close": [100.0] * 20,
            }
        )
        result = simulate(
            predictions,
            prices,
            mode="long-short",
            horizon=1,
            tail_fraction=0.20,
            short_fraction=0.05,
            cost_bps=0.0,
        )
        self.assertEqual(result["long_names"].iloc[0], 4)
        self.assertEqual(result["short_names"].iloc[0], 1)
        self.assertAlmostEqual(result["net_exposure"].iloc[0], 0.0, places=12)

    def test_long_only_dynamic_threshold_does_not_require_short_cutoff(self):
        day = pd.Timestamp("2025-01-02")
        predictions = pd.DataFrame(
            {
                "Date": [day] * 20,
                "target_end_date": [day] * 20,
                "Ticker": [f"T{i:02d}" for i in range(20)],
                "prediction": np.linspace(-3.0, 3.0, 20),
            }
        )
        prices = pd.DataFrame(
            {
                "Date": [day] * 20,
                "Ticker": [f"T{i:02d}" for i in range(20)],
                "Open": [100.0] * 20,
                "Close": [100.0] * 20,
            }
        )
        result = simulate(
            predictions,
            prices,
            mode="long-only",
            horizon=1,
            tail_fraction=0.10,
            long_z=1.0,
            cost_bps=0.0,
        )
        self.assertTrue(result["signal_entered"].iloc[0])
        self.assertGreater(result["long_names"].iloc[0], 0)
        self.assertEqual(result["short_names"].iloc[0], 0)


if __name__ == "__main__":
    unittest.main()
