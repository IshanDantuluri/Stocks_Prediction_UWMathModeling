import csv
import json
import sqlite3
import tempfile
import unittest
from datetime import date
from pathlib import Path

from news_reasoning import (
    AnalysisResult,
    Entity,
    Event,
    EventAssessment,
    aggregate_assessments,
    export_trading_features,
    initialize_memory,
    process_event,
    read_state,
)


class FakeProvider:
    model_id = "fake-model"
    prompt_version = "test-v1"

    def __init__(self):
        self.requests = []

    def analyze(self, request):
        self.requests.append(request)
        return AnalysisResult(
            EventAssessment(
                news_signed_impact=-0.6,
                news_confidence=0.8,
                news_novelty=0.7,
                news_persistence=0.9,
                news_uncertainty_change=0.4,
                category_impacts={"regulatory_impact": -0.75},
                channel_impacts={"revenue_channel_impact": -0.5},
                reported_fact_count=4,
                analysis_count=2,
                speculation_count=1,
                thread_key="export-controls",
            ),
            [{"thread_key": "export-controls", "status": "active"}],
            "Export restrictions remain an active risk.",
        )


class NewsReasoningTests(unittest.TestCase):
    def test_assessment_validation(self):
        with self.assertRaises(ValueError):
            EventAssessment(1.1, 0.5, 0.5, 0.5)
        with self.assertRaises(ValueError):
            EventAssessment(
                0, 0.5, 0.5, 0.5, category_impacts={"unknown": 1}
            )

    def test_daily_aggregation_counts_and_nullable_impacts(self):
        one = EventAssessment(
            -0.8,
            0.9,
            0.8,
            0.7,
            category_impacts={"regulatory_impact": -0.9},
            reported_fact_count=3,
        )
        two = EventAssessment(
            0.4,
            0.5,
            0.2,
            0.3,
            category_impacts={"product_impact": 0.6},
            reported_fact_count=2,
            speculation_count=1,
        )
        result = aggregate_assessments(
            [one, two], [1, 2, 2], ["a.com", "b.com", "a.com"]
        )
        self.assertEqual(result["news_article_count"], 2)
        self.assertEqual(result["news_unique_event_count"], 2)
        self.assertEqual(result["news_source_count"], 2)
        self.assertEqual(result["reported_fact_count"], 5)
        self.assertEqual(result["regulatory_impact"], -0.9)
        self.assertEqual(result["product_impact"], 0.6)
        self.assertGreater(result["news_disagreement"], 0)

    def test_chronological_processing_is_cached_and_provenanced(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory.sqlite3"
            event = Event(
                "event-1",
                date(2024, 1, 3),
                "New export restrictions",
                "Restrictions affect accelerator sales.",
                (10, 11),
                ("a.com", "b.com"),
            )
            entity = Entity("ticker", "NVDA")
            provider = FakeProvider()
            with sqlite3.connect(path) as db:
                initialize_memory(db)
                self.assertTrue(process_event(db, event, entity, provider, None))
                self.assertFalse(process_event(db, event, entity, provider, None))
                state = read_state(
                    db, entity, provider.model_id, provider.prompt_version
                )
                self.assertEqual(state.as_of, event.event_date)
                self.assertEqual(len(provider.requests), 1)
                features_json, events_json, articles_json = db.execute(
                    """SELECT features_json,event_ids_json,article_ids_json
                       FROM daily_features"""
                ).fetchone()
                self.assertEqual(
                    json.loads(features_json)["news_article_count"], 2
                )
                self.assertEqual(json.loads(events_json), ["event-1"])
                self.assertEqual(json.loads(articles_json), [10, 11])
                cutoff = db.execute(
                    "SELECT exclusive_cutoff FROM retrieval_contexts"
                ).fetchone()[0]
                self.assertEqual(cutoff, "2024-01-03")

    def test_rejects_processing_older_than_current_state(self):
        provider = FakeProvider()
        with sqlite3.connect(":memory:") as db:
            initialize_memory(db)
            entity = Entity("sector", "Semiconductors")
            process_event(
                db,
                Event("new", date(2024, 2, 2), "a", "b", (1,)),
                entity,
                provider,
                None,
            )
            with self.assertRaises(ValueError):
                process_event(
                    db,
                    Event("old", date(2024, 2, 1), "a", "b", (2,)),
                    entity,
                    provider,
                    None,
                )

    def test_news_is_available_only_at_first_strictly_later_session(self):
        provider = FakeProvider()
        entity = Entity("ticker", "NVDA")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "features.csv"
            with sqlite3.connect(":memory:") as db:
                initialize_memory(db)
                for event_id, event_day in (
                    ("fri", date(2024, 1, 5)),
                    ("sat", date(2024, 1, 6)),
                    ("sun", date(2024, 1, 7)),
                    ("mon", date(2024, 1, 8)),
                ):
                    process_event(
                        db,
                        Event(event_id, event_day, "a", "b", (len(event_id),)),
                        entity,
                        provider,
                        None,
                    )
                count, deferred = export_trading_features(
                    db,
                    output,
                    [
                        date(2024, 1, 5),
                        date(2024, 1, 8),
                        date(2024, 1, 9),
                    ],
                )
            with output.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual((count, deferred), (2, 0))
            self.assertEqual(
                [row["trade_date"] for row in rows],
                ["2024-01-08", "2024-01-09"],
            )
            self.assertEqual(
                [int(row["news_unique_event_count"]) for row in rows], [3, 1]
            )
            self.assertEqual(rows[0]["first_news_date"], "2024-01-05")
            self.assertEqual(rows[0]["last_news_date"], "2024-01-07")


if __name__ == "__main__":
    unittest.main()
