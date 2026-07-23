import unittest

import numpy as np
import pandas as pd

import quant_boosted_baseline as boosted


class BoostedBaselineTests(unittest.TestCase):
    def test_lagged_features_end_before_trade_date(self):
        dates = pd.bdate_range("2022-01-03", periods=35)
        rows = []
        for ticker, sector, offset in (
            ("AAA", "Technology", 0.0),
            ("BBB", "Financials", 2.0),
        ):
            for index, date in enumerate(dates):
                close = 100.0 + offset + index
                rows.append({
                    "Date": date,
                    "Ticker": ticker,
                    "Open": close - 0.5,
                    "High": close + index / 10,
                    "Low": close - 1.0,
                    "Close": close,
                    "Volume": 1_000 + index,
                })
        prices = pd.DataFrame(rows)
        tickers = pd.DataFrame({
            "Symbol": ["AAA", "BBB"],
            "GICS Sector": ["Technology", "Financials"],
        })
        market = pd.DataFrame({
            "Date": dates,
            "Open": np.full(len(dates), 100.0),
            "Close": np.full(len(dates), 100.0),
        })

        frame, features, _ = boosted.build_tabular_frame(
            prices, tickers, market, lags=(1, 5, 20), horizon=5
        )
        trade_date = dates[25]
        row = frame[
            (frame["Ticker"] == "AAA") & (frame["Date"] == trade_date)
        ].iloc[0]
        previous = prices[
            (prices["Ticker"] == "AAA")
            & (prices["Date"] == dates[24])
        ].iloc[0]
        expected_range = (
            (previous["High"] - previous["Low"]) / previous["Close"]
        )
        self.assertAlmostEqual(
            row["intraday_range__lag1"], expected_range
        )
        current = prices[
            (prices["Ticker"] == "AAA")
            & (prices["Date"] == trade_date)
        ].iloc[0]
        exit_session = prices[
            (prices["Ticker"] == "AAA")
            & (prices["Date"] == dates[29])
        ].iloc[0]
        self.assertAlmostEqual(
            row["target_alpha"],
            exit_session["Close"] / current["Open"] - 1.0,
        )
        self.assertEqual(row["target_end_date"], dates[29])
        self.assertEqual(len(features), 61)

    def test_daily_cross_sectional_metrics(self):
        actual = np.array([-0.02, -0.01, 0.01, 0.02] * 2)
        predictions = actual.copy()
        dates = np.array(
            [np.datetime64("2025-01-02")] * 4
            + [np.datetime64("2025-01-03")] * 4
        )
        ranks = np.array([-0.375, -0.125, 0.125, 0.375] * 2)
        metrics = boosted.evaluate_predictions(
            actual,
            predictions,
            dates,
            regression_targets=ranks,
            hac_lags=4,
        )
        self.assertAlmostEqual(metrics["mean_daily_ic"], 1.0)
        self.assertAlmostEqual(metrics["half_accuracy"], 1.0)
        self.assertEqual(metrics["daily_ic_days"], 2)

    def test_matched_ablation_feature_contracts(self):
        quant = ["return__lag1", "sector_code"]
        news = list(boosted.SCOPED_LLM_FEATURE_NAMES)

        self.assertEqual(
            boosted.select_model_features(quant, "quant-only"), quant
        )
        self.assertEqual(
            boosted.select_model_features(quant, "news-only"), news
        )
        self.assertEqual(
            boosted.select_model_features(quant, "quant-news"),
            quant + news,
        )
        self.assertEqual(len(news), 72)
        with self.assertRaises(ValueError):
            boosted.select_model_features(quant, "unknown")


if __name__ == "__main__":
    unittest.main()
