import csv
import json
import sqlite3
import tempfile
import unittest
from datetime import date
from pathlib import Path

import numpy as np

from news_reasoning import CATEGORY_FIELDS, CHANNEL_FIELDS, initialize_memory
from sec_distill import decision_trade_date, merge_model_features, predict, train
from sec_features import initialize


class SecDistillTest(unittest.TestCase):
    def test_merges_deterministic_and_llm_rows_into_factor_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            deterministic = root / "det.csv"
            llm = root / "llm.csv"
            output = root / "model.csv"
            deterministic.write_text(
                "trade_date,ticker,sec_filing_count\n2025-01-06,TEST,2\n"
            )
            llm.write_text(
                "trade_date,entity_id,scope,news_signed_impact\n"
                "2025-01-06,TEST,ticker,0.4\n"
            )
            self.assertEqual(
                merge_model_features(deterministic, output, llm),
                (1, 2),
            )
            with output.open() as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["factor__sec_filing_count"], "2")
            self.assertEqual(
                row["factor__sec_llm__news_signed_impact"], "0.4"
            )

    def test_after_close_filing_waits_for_the_following_close(self):
        sessions = [
            date(2025, 1, 3),
            date(2025, 1, 6),
            date(2025, 1, 7),
            date(2025, 1, 8),
        ]
        self.assertEqual(
            decision_trade_date("2025-01-03T20:00:00Z", sessions),
            date(2025, 1, 6),
        )
        self.assertEqual(
            decision_trade_date("2025-01-03T21:05:00Z", sessions),
            date(2025, 1, 7),
        )

    def test_trains_chronologically_and_predicts_all_feature_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            features_path = root / "features.sqlite3"
            reasoning_path = root / "reasoning.sqlite3"
            model_path = root / "distiller.joblib"
            report_path = root / "report.json"
            output_path = root / "predictions.csv"

            with sqlite3.connect(features_path) as db:
                initialize(db)
                for index in range(30):
                    year = 2020 if index < 20 else 2023 if index < 25 else 2025
                    event_id = f"sec:event-{index}"
                    vector = np.asarray(
                        [index / 30, 1 - index / 30], dtype="<f2"
                    ).tobytes()
                    db.execute(
                        """
                        INSERT INTO sec_event_features(
                            event_id,accession,cik,filing_date,accepted_at,form,
                            item_codes_json,importance_score,after_market_close,
                            accepted_hour_et,document_count,exhibit99_count,
                            primary_count,press_release_count,char_count,word_count,
                            number_count,currency_count,percent_count,positive_count,
                            negative_count,uncertainty_count,litigation_count,
                            constraining_count,embedding_config_id,dimension,vector
                        ) VALUES(?,?,?,?,?,'8-K','["2.02"]',5,1,16.1,2,1,1,1,
                                 1000,200,10,2,1,3,1,2,0,1,1,2,?)
                        """,
                        (
                            event_id,
                            f"event-{index}",
                            index + 1,
                            f"{year}-01-01",
                            f"{year}-01-01T21:05:00Z",
                            vector,
                        ),
                    )
                    db.execute(
                        """
                        INSERT INTO sec_event_entities(
                            event_id,ticker,company_name,sector,
                            days_since_previous,embedding_novelty
                        ) VALUES(?,?,'Test','Industrials',30,0.2)
                        """,
                        (event_id, f"T{index:02d}"),
                    )
                db.commit()

            with sqlite3.connect(reasoning_path) as db:
                initialize_memory(db)
                for index in range(30):
                    year = 2020 if index < 20 else 2023 if index < 25 else 2025
                    event_id = f"sec:event-{index}"
                    ticker = f"T{index:02d}"
                    assessment = {
                        "news_signed_impact": 2 * index / 29 - 1,
                        "news_confidence": 0.8,
                        "news_novelty": 0.2,
                        "news_persistence": 0.5,
                        "news_uncertainty_change": 0.0,
                        "news_disagreement": 0.1,
                        "category_impacts": {
                            name: (0.3 if name == "earnings_impact" else None)
                            for name in CATEGORY_FIELDS
                        },
                        "channel_impacts": {
                            name: (0.2 if name == "revenue_channel_impact" else None)
                            for name in CHANNEL_FIELDS
                        },
                        "reported_fact_count": 4,
                        "analysis_count": 1,
                        "speculation_count": 0,
                    }
                    db.execute(
                        """
                        INSERT INTO events(
                            event_id,event_date,title,summary,article_ids_json,
                            source_domains_json,retrieval_query
                        ) VALUES(?,?,?,'summary','[1]','["sec.gov"]',NULL)
                        """,
                        (event_id, f"{year}-01-01", event_id),
                    )
                    db.execute(
                        """
                        INSERT INTO assessments(
                            event_id,scope,entity_id,model_id,prompt_version,
                            assessment_json
                        ) VALUES(?,'ticker',?,'deepseek-v4-flash',
                                 'sec-reasoning-v1',?)
                        """,
                        (event_id, ticker, json.dumps(assessment)),
                    )
                db.commit()

            report = train(
                features_path,
                reasoning_path,
                model_path,
                report_path,
                "deepseek-v4-flash",
                "sec-reasoning-v1",
                [0.1, 1.0],
            )
            self.assertEqual(
                report["split_counts"],
                {"train": 20, "validation": 5, "test": 5},
            )
            self.assertEqual(
                predict(features_path, model_path, output_path),
                30,
            )
            with output_path.open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 30)
            self.assertEqual(rows[0]["feature_source"], "distilled_deepseek")
            self.assertEqual(rows[0]["news_article_count"], "2")


if __name__ == "__main__":
    unittest.main()
