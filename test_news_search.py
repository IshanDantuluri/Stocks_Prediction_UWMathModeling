import json
import sqlite3
import tempfile
import unittest
from datetime import date
from pathlib import Path

import numpy as np

from news_search import ExactNewsIndex, hydrate_hits


class NewsSearchTests(unittest.TestCase):
    def test_exact_search_deduplicates_articles_and_enforces_cutoff(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            np.save(path / "vectors.npy", np.array([
                [1, 0], [.9, .1], [.8, .2], [0, 1]
            ], dtype=np.float16))
            np.save(path / "chunk_ids.npy", np.array([10, 11, 12, 13]))
            np.save(path / "article_ids.npy", np.array([1, 1, 2, 3]))
            np.save(path / "event_dates.npy", np.array([
                date(2020, 1, 1).toordinal(), date(2020, 1, 1).toordinal(),
                date(2025, 1, 1).toordinal(), 0
            ], dtype=np.int32))
            (path / "manifest.json").write_text(json.dumps({
                "count": 4, "dimension": 2, "model_name": "test"
            }))
            index = ExactNewsIndex(path)
            hits = index.search_vector(
                np.array([1, 0]), top_articles=5, before=date(2021, 1, 1),
                candidate_chunks=4, block_size=2,
            )
            self.assertEqual([hit.article_id for hit in hits], [1])
            self.assertEqual(hits[0].chunk_id, 10)

    def test_keyword_query_removes_stopwords_and_deduplicates(self):
        self.assertEqual(
            ExactNewsIndex._fts_query("The NVIDIA export export restrictions"),
            '"nvidia" AND "export" AND "restrictions"',
        )

    def test_hydration_includes_duplicate_event_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "archive.sqlite3"
            with sqlite3.connect(archive) as db:
                db.executescript(
                    """
                    CREATE TABLE articles(id INTEGER PRIMARY KEY,title TEXT,domain TEXT,
                        source_url TEXT,published_at TEXT,canonical_article_id INTEGER);
                    CREATE TABLE events(article_id INTEGER,date TEXT,event_category TEXT);
                    CREATE TABLE article_chunks(id INTEGER PRIMARY KEY,body_text TEXT);
                    INSERT INTO articles VALUES
                        (1,'Title','example.com','https://example.com/a',NULL,1),
                        (2,'Copy','other.com','https://other.com/a',NULL,1);
                    INSERT INTO events VALUES (2,'2020-01-02','product_launch');
                    INSERT INTO article_chunks VALUES (10,'Relevant passage');
                    """
                )
            from news_search import SearchHit
            result = hydrate_hits(archive, [SearchHit(1, .9, 10)])
            self.assertEqual(result[0].first_event_date, "2020-01-02")
            self.assertEqual(result[0].event_categories, "product_launch")
            self.assertEqual(result[0].passage, "Relevant passage")


if __name__ == "__main__":
    unittest.main()
