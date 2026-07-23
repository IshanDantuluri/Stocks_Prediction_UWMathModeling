import csv
import sqlite3
import tempfile
import unittest
from datetime import date
from pathlib import Path

import numpy as np

from news_events import (
    ArticleRecord,
    Candidate,
    candidates_for_text,
    cluster_articles,
    import_entities,
    import_alias_overrides,
    initialize,
    load_saved_candidates,
    persist_day,
    save_candidates,
    verification_payload,
)


class NewsEventsTests(unittest.TestCase):
    def test_same_day_clustering_uses_similarity_and_title_overlap(self):
        day = date(2024, 1, 3)
        articles = [
            ArticleRecord(1, day, "Nvidia faces new export restrictions", "a.com",
                          np.array([1.0, 0.0, 0.0])),
            ArticleRecord(2, day, "New Nvidia export restrictions announced", "b.com",
                          np.array([0.99, 0.05, 0.0])),
            ArticleRecord(3, day, "Bank reports quarterly earnings", "c.com",
                          np.array([0.0, 1.0, 0.0])),
        ]
        clusters = cluster_articles(articles)
        self.assertEqual([item.article_ids for item in clusters], [(1, 2), (3,)])
        self.assertTrue(np.isclose(np.linalg.norm(clusters[0].centroid), 1.0))

    def test_clustering_rejects_multiple_dates(self):
        with self.assertRaises(ValueError):
            cluster_articles([
                ArticleRecord(1, date(2024, 1, 1), "a", "a", np.array([1, 0])),
                ArticleRecord(2, date(2024, 1, 2), "a", "a", np.array([1, 0])),
            ])

    def test_entity_import_alias_candidates_and_validity(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "entities.csv"
            with source.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow([
                    "ticker", "company_name", "GICS Sector", "industry",
                    "aliases", "valid_from", "valid_to",
                ])
                writer.writerow([
                    "NVDA", "NVIDIA Corporation", "Technology", "Semiconductors",
                    "Nvidia;GeForce", "2010-01-01", "",
                ])
                writer.writerow([
                    "META", "Meta Platforms Inc.", "Communication Services",
                    "Interactive Media", "Facebook", "2021-10-28", "",
                ])
                writer.writerow([
                    "GOOGL", "Alphabet Inc. (Class A)", "Communication Services",
                    "Interactive Media", "", "2015-01-01", "",
                ])
                writer.writerow([
                    "A", "Agilent Technologies Inc.", "Health Care",
                    "Life Sciences Tools", "", "2000-01-01", "",
                ])
            with sqlite3.connect(":memory:") as db:
                entities, aliases = import_entities(db, source)
                self.assertEqual(entities, 4)
                self.assertGreaterEqual(aliases, 6)
                candidates = candidates_for_text(
                    db,
                    "Nvidia said its GeForce products were affected.",
                    date(2020, 1, 1),
                )
                self.assertEqual(candidates[0].entity_id, "NVDA")
                self.assertIn("Technology", [item.entity_id for item in candidates])
                old_meta = candidates_for_text(
                    db, "Facebook announced a change.", date(2020, 1, 1)
                )
                self.assertNotIn("META", [item.entity_id for item in old_meta])
                alphabet = candidates_for_text(
                    db, "Alphabet announced a change.", date(2020, 1, 1)
                )
                self.assertIn("GOOGL", [item.entity_id for item in alphabet])
                ordinary_letters = candidates_for_text(
                    db, "A company made a routine filing.", date(2020, 1, 1)
                )
                self.assertNotIn("A", [item.entity_id for item in ordinary_letters])

                overrides = Path(directory) / "aliases.csv"
                with overrides.open("w", newline="") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(["ticker", "alias", "valid_from", "valid_to"])
                    writer.writerow(["GOOGL", "Google", "2015-01-01", ""])
                    writer.writerow(["MISSING", "Unknown", "", ""])
                imported, skipped = import_alias_overrides(db, overrides)
                self.assertEqual((imported, skipped), (1, 1))
                google = candidates_for_text(
                    db, "Google announced a change.", date(2020, 1, 1)
                )
                self.assertIn("GOOGL", [item.entity_id for item in google])

    def test_persist_day_is_resumable(self):
        day = date(2024, 1, 3)
        articles = [
            ArticleRecord(1, day, "Same event", "a.com", np.array([1.0, 0.0])),
            ArticleRecord(2, day, "Same event update", "b.com", np.array([0.99, 0.01])),
        ]
        with sqlite3.connect(":memory:") as db:
            initialize(db)
            db.execute(
                """INSERT INTO cluster_configs VALUES
                   (1,'hash',1,.84,.92,.18,'{}',CURRENT_TIMESTAMP)"""
            )
            first = persist_day(db, 1, "hash", articles, (.84, .92, .18))
            second = persist_day(db, 1, "hash", articles, (.84, .92, .18))
            self.assertEqual(first, (1, True))
            self.assertEqual(second, (1, False))

    def test_verification_contract_includes_candidates(self):
        payload = verification_payload(
            "cluster", date(2024, 1, 1), "Title", "Text", []
        )
        self.assertIn("additional_company_names", payload["output"])
        self.assertIn("event_summary", payload["output"])

    def test_saved_candidates_can_be_reused(self):
        candidates = [
            Candidate("ticker", "NVDA", 1.0, ("company match: Nvidia",)),
            Candidate(
                "sector",
                "Technology",
                0.65,
                ("inherited from ticker candidate NVDA",),
            ),
        ]
        with sqlite3.connect(":memory:") as db:
            initialize(db)
            db.execute(
                """INSERT INTO cluster_configs VALUES
                   (1,'hash',1,.84,.92,.18,'{}',CURRENT_TIMESTAMP)"""
            )
            db.execute(
                """INSERT INTO event_clusters(
                     cluster_id,config_id,event_date,representative_title,
                     centroid,dimension,article_count,source_count
                   ) VALUES ('cluster',1,'2024-01-01','Title',X'00',1,1,1)"""
            )
            save_candidates(db, "cluster", candidates)
            self.assertEqual(load_saved_candidates(db, "cluster"), candidates)


if __name__ == "__main__":
    unittest.main()
