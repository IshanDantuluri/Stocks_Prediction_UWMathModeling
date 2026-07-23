import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import mathmodellingstocksgrumodel as model


class StockModelDataContractTests(unittest.TestCase):
    def _quant_frame(self):
        dates = pd.bdate_range("2022-11-01", periods=80)
        rows = []
        for ticker, sector, offset in (
            ("AAA", "Information Technology", 0.0),
            ("BBB", "Financials", 2.0),
        ):
            for index, day in enumerate(dates):
                close = 100.0 + offset + index * 0.1
                rows.append({
                    "Date": day,
                    "Ticker": ticker,
                    "Sector": sector,
                    "Open": close - 0.05,
                    "High": close + 0.2,
                    "Low": close - 0.2,
                    "Close": close,
                    "Volume": 1_000 + index,
                })
        frame = pd.DataFrame(rows).sort_values(["Ticker", "Date"])
        frame = model.engineer_quant_features(frame)
        frame = model.engineer_cross_sectional_features(frame)
        frame["target_alpha_1d"] = frame["raw_target_1d"]
        return frame

    def test_trade_date_sample_uses_only_prior_price_sessions(self):
        frame = self._quant_frame()
        tickers = pd.DataFrame({
            "Symbol": ["AAA", "BBB"],
            "GICS Sector": ["Information Technology", "Financials"],
        })
        merged = model.merge_scoped_news_data(frame, tickers)
        store = model.StockSequenceStore(
            merged, model.QUANT_FEATURE_NAMES, model.SCOPED_LLM_FEATURE_NAMES
        )
        dataset = model.MultiStockDataset(
            store, start="2023-01-01", seq_len=30
        )

        group_index, row_index = dataset.samples[0]
        group = store.groups[group_index]
        self.assertEqual(group["dates"][row_index], np.datetime64("2023-01-02"))
        self.assertEqual(
            group["dates"][row_index - 1], np.datetime64("2022-12-30")
        )
        quant, news, target, trade_day = dataset[0]
        self.assertEqual(quant.shape, (30, len(model.QUANT_FEATURE_NAMES)))
        self.assertEqual(news.shape, (len(model.SCOPED_LLM_FEATURE_NAMES),))
        self.assertEqual(target.shape, (1,))
        self.assertEqual(
            trade_day, np.datetime64("2023-01-02").astype(np.int64)
        )

    def test_scoped_news_join_and_count_mapping(self):
        frame = self._quant_frame().iloc[[60, 140]].copy()
        tickers = pd.DataFrame({
            "Symbol": ["AAA", "BBB"],
            "GICS Sector": ["Information Technology", "Financials"],
        })
        trade_date = frame["Date"].iloc[0].strftime("%Y-%m-%d")
        frame.loc[:, "Date"] = pd.Timestamp(trade_date)
        zeros = {feature: 0.0 for feature in model.LLM_FEATURE_NAMES}
        rows = []
        for scope, entity, impact, count in (
            ("ticker", "AAA", 0.7, 5),
            ("sector", "Information Technology", 0.4, 2),
            ("market", "US", -0.2, 10),
        ):
            rows.append({
                "trade_date": trade_date,
                "scope": scope,
                "entity_id": entity,
                "model_id": "test-model",
                "prompt_version": "test-prompt",
                **zeros,
                "news_signed_impact": impact,
                "news_article_count": count,
            })
        news = pd.DataFrame(rows)

        with patch.object(model, "_read_news_features", return_value=news):
            merged = model.merge_scoped_news_data(
                frame,
                tickers,
                news_path="unused.csv",
                model_id="test-model",
                prompt_version="test-prompt",
            )

        aaa = merged[merged["Ticker"] == "AAA"].iloc[0]
        bbb = merged[merged["Ticker"] == "BBB"].iloc[0]
        self.assertEqual(aaa["ticker__news_signed_impact"], 0.7)
        self.assertEqual(aaa["sector__news_signed_impact"], 0.4)
        self.assertEqual(bbb["sector__news_signed_impact"], 0.0)
        self.assertEqual(aaa["market__news_signed_impact"], -0.2)
        self.assertEqual(aaa["ticker__news_article_count"], 1.0)
        self.assertEqual(aaa["sector__news_article_count"], 0.4)
        self.assertEqual(aaa["market__news_article_count"], 1.0)


if __name__ == "__main__":
    unittest.main()
