import gzip
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from prospective_news import (
    clean_prospective_archive,
    initialize,
    materialize,
    parse_alpha_time,
    redact_secrets,
    upsert_alpha_feed,
    write_raw_payload,
)


class ProspectiveNewsTests(unittest.TestCase):
    def sample(self, body=None):
        item = {
            "title": "Acme reports quarterly results",
            "url": "https://example.com/acme-results",
            "time_published": "20260723T143000",
            "source": "Example",
            "summary": "A short provider abstract.",
            "ticker_sentiment": [
                {
                    "ticker": "ACME",
                    "relevance_score": "0.91",
                    "ticker_sentiment_score": "0.22",
                    "ticker_sentiment_label": "Somewhat-Bullish",
                }
            ],
        }
        if body is not None:
            item["body"] = body
        return item

    def test_alpha_timestamp_is_explicit_utc(self):
        self.assertEqual(
            parse_alpha_time("20260723T143000"), "2026-07-23T14:30:00+00:00"
        )
        self.assertEqual(
            redact_secrets("url?apikey=secret-value&limit=1"),
            "url?apikey=REDACTED&limit=1",
        )

    def test_raw_retention_and_upsert_preserve_first_retrieval(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with sqlite3.connect(root / "news.sqlite3") as db:
                initialize(db)
                run = db.execute(
                    """INSERT INTO provider_runs(
                         provider,started_at,status,request_parameters_json)
                       VALUES ('alpha_vantage','2026-07-23T15:00:00+00:00',
                               'running','{}')"""
                ).lastrowid
                payload = {"feed": [self.sample()]}
                payload_id = write_raw_payload(
                    db, root / "raw", run, "alpha_vantage", payload,
                    "2026-07-23T15:00:00+00:00",
                )
                upsert_alpha_feed(
                    db, payload["feed"], "2026-07-23T15:00:00+00:00", payload_id
                )
                upsert_alpha_feed(
                    db, payload["feed"], "2026-07-23T16:00:00+00:00", payload_id
                )
                row = db.execute(
                    """SELECT first_retrieved_at,last_retrieved_at,published_at
                       FROM articles"""
                ).fetchone()
                self.assertEqual(row[0], "2026-07-23T15:00:00+00:00")
                self.assertEqual(row[1], "2026-07-23T16:00:00+00:00")
                self.assertEqual(row[2], "2026-07-23T14:30:00+00:00")
                raw_path = Path(
                    db.execute("SELECT raw_path FROM raw_payloads").fetchone()[0]
                )
                with gzip.open(raw_path, "rt", encoding="utf-8") as handle:
                    self.assertEqual(json.load(handle), payload)

    def test_materialize_records_availability_and_cleans_body(self):
        body = " ".join(
            [
                f"Business segment {index} reported revenue growth and a higher "
                "operating margin for the latest fiscal quarter while management "
                "described customer demand and expected capital investment."
                for index in range(12)
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source.sqlite3"
            with sqlite3.connect(source_path) as db:
                initialize(db)
                run = db.execute(
                    """INSERT INTO provider_runs(
                         provider,started_at,status,request_parameters_json)
                       VALUES ('alpha_vantage','2026-07-23T15:00:00+00:00',
                               'running','{}')"""
                ).lastrowid
                payload = {"feed": [self.sample(body)]}
                payload_id = write_raw_payload(
                    db, root / "raw", run, "alpha_vantage", payload,
                    "2026-07-23T15:00:00+00:00",
                )
                upsert_alpha_feed(
                    db, payload["feed"], "2026-07-23T15:00:00+00:00", payload_id
                )
                db.commit()
            archive = root / "archive.sqlite3"
            self.assertEqual(materialize(source_path, archive), (1, 1))
            self.assertEqual(materialize(source_path, archive), (1, 1))
            with sqlite3.connect(archive) as db:
                article = db.execute(
                    "SELECT status,quality_status,effective_date FROM articles"
                ).fetchone()
                self.assertEqual(article[0], "ok")
                self.assertEqual(article[1], "usable")
                self.assertEqual(article[2], "2026-07-23")
                source = db.execute(
                    """SELECT available_at,tagged_tickers_json
                       FROM prospective_article_sources"""
                ).fetchone()
                self.assertEqual(source[0], "2026-07-23T15:00:00+00:00")
                self.assertEqual(json.loads(source[1]), ["ACME"])
                self.assertEqual(
                    db.execute("SELECT COUNT(*) FROM events").fetchone()[0], 1
                )

    def test_url_only_article_can_be_cleaned_after_fetch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source.sqlite3"
            with sqlite3.connect(source_path) as db:
                initialize(db)
                run = db.execute(
                    """INSERT INTO provider_runs(
                         provider,started_at,status,request_parameters_json)
                       VALUES ('alpha_vantage','2026-07-23T15:00:00+00:00',
                               'running','{}')"""
                ).lastrowid
                payload = {"feed": [self.sample()]}
                payload_id = write_raw_payload(
                    db, root / "raw", run, "alpha_vantage", payload,
                    "2026-07-23T15:00:00+00:00",
                )
                upsert_alpha_feed(
                    db, payload["feed"], "2026-07-23T15:00:00+00:00", payload_id
                )
                db.commit()
            archive = root / "archive.sqlite3"
            materialize(source_path, archive)
            body = " ".join(
                ["Detailed financial results and forward business guidance."] * 30
            )
            with sqlite3.connect(archive) as db:
                db.execute(
                    """UPDATE articles SET status='ok',article_text_raw=?,
                       article_text=?""",
                    (body, body),
                )
            self.assertEqual(clean_prospective_archive(archive), 1)
            self.assertEqual(clean_prospective_archive(archive), 0)
            with sqlite3.connect(archive) as db:
                self.assertEqual(
                    db.execute("SELECT quality_status FROM articles").fetchone()[0],
                    "usable",
                )


if __name__ == "__main__":
    unittest.main()
