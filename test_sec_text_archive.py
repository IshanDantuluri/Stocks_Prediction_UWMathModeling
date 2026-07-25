import sqlite3
import tempfile
import unittest
from pathlib import Path

from sec_text_archive import (
    choose_documents,
    connect_archive,
    discover_filings,
    fetch_documents,
    plan_filings,
    process_archive,
)


INDEX_HTML = b"""
<html><body>
<table class="tableFile" summary="Document Format Files">
  <tr><th>Seq</th><th>Description</th><th>Document</th><th>Type</th><th>Size</th></tr>
  <tr><td>1</td><td>Current report</td><td><a href="acme-8k.htm">acme-8k.htm</a></td><td>8-K</td><td>12000</td></tr>
  <tr><td>2</td><td>Press Release</td><td><a href="ex991.htm">ex991.htm</a></td><td>EX-99.1</td><td>8000</td></tr>
  <tr><td>3</td><td>Graphic</td><td><a href="logo.jpg">logo.jpg</a></td><td>GRAPHIC</td><td>1000</td></tr>
  <tr><td>4</td><td>Material contract</td><td><a href="ex101.htm">ex101.htm</a></td><td>EX-10.1</td><td>9000</td></tr>
</table>
</body></html>
"""


def article_html(label: str) -> bytes:
    topics = (
        "revenue", "operations", "management", "customers", "suppliers", "liquidity",
        "investment", "guidance", "competition", "regulation", "strategy", "outlook",
    )
    paragraphs = "".join(
        f"<p>{label} discusses {topic} and reports detailed financial information "
        "about business conditions, risks, expectations, decisions, and plans for "
        "the coming reporting period using evidence specific to this topic.</p>"
        for topic in topics
    )
    return f"<html><head><title>{label}</title></head><body>{paragraphs}</body></html>".encode()


class FakeClient:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def get(self, url):
        self.calls.append(url)
        return self.responses[url]


class SelectionTests(unittest.TestCase):
    def test_selects_primary_and_exhibit_99_but_not_unrelated_exhibits(self):
        choices = choose_documents(INDEX_HTML, "acme-8k.htm")
        self.assertEqual(
            [(choice.document_name, choice.selection_reason) for choice in choices],
            [("acme-8k.htm", "primary_document"), ("ex991.htm", "exhibit_99")],
        )


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.archive = connect_archive(self.root / "archive.sqlite3")
        self.source = sqlite3.connect(":memory:")
        self.source.executescript(
            """
            CREATE TABLE sec_filings(
              accession TEXT PRIMARY KEY,cik INTEGER,filing_date TEXT,
              accepted_at TEXT,available_at_quality TEXT,form TEXT,
              primary_document TEXT,retrieved_at TEXT
            );
            CREATE TABLE entities(
              ticker TEXT,cik INTEGER,company_name TEXT,sector TEXT
            );
            INSERT INTO entities VALUES(123456,'ACME','Acme Corporation','Industrials');
            """
            .replace("VALUES(123456,'ACME'", "VALUES('ACME',123456")
        )

    def tearDown(self):
        self.source.close()
        self.archive.close()
        self.temporary.cleanup()

    def insert_filing(self, accession, accepted_at="2025-02-03T21:04:05+00:00"):
        self.source.execute(
            "INSERT INTO sec_filings VALUES(?,?,?,?,?,?,?,?)",
            (
                accession, 123456, "2025-02-03", accepted_at,
                "sec_acceptance_datetime", "8-K", "acme-8k.htm",
                "2026-07-23T00:00:00+00:00",
            ),
        )
        self.source.commit()

    def test_plan_preserves_exact_acceptance_and_is_resumable(self):
        self.insert_filing("0000123456-25-000001")
        first = plan_filings(
            self.archive, self.source, "2025-01-01", None, ("8-K",), (), None, True
        )
        second = plan_filings(
            self.archive, self.source, "2025-01-01", None, ("8-K",), (), None, True
        )
        self.assertEqual((first, second), (1, 0))
        row = self.archive.execute(
            "SELECT accepted_at,available_at_quality FROM sec_filing_queue"
        ).fetchone()
        self.assertEqual(
            row, ("2025-02-03T21:04:05+00:00", "sec_acceptance_datetime")
        )

    def test_bounded_planning_advances_past_existing_filings(self):
        for suffix in range(1, 4):
            self.insert_filing(f"0000123456-25-{suffix:06d}")
        first = plan_filings(
            self.archive, self.source, "2025-01-01", None,
            ("8-K",), (), 2, True,
        )
        second = plan_filings(
            self.archive, self.source, "2025-01-01", None,
            ("8-K",), (), 2, True,
        )
        self.assertEqual((first, second), (2, 1))
        self.assertEqual(
            self.archive.execute("SELECT COUNT(*) FROM sec_filing_queue").fetchone()[0],
            3,
        )

    def test_fetch_resume_dedup_and_cleaning(self):
        accessions = ("0000123456-25-000001", "0000123456-25-000002")
        for accession in accessions:
            self.insert_filing(accession)
        plan_filings(
            self.archive, self.source, "2025-01-01", None, ("8-K",), (), None, True
        )
        index_responses = {
            row[0]: (INDEX_HTML, 200, "")
            for row in self.archive.execute("SELECT index_url FROM sec_filing_queue")
        }
        index_client = FakeClient(index_responses)
        raw_dir = self.root / "raw"
        discovered, failed = discover_filings(
            self.archive, index_client, raw_dir, None, workers=4
        )
        self.assertEqual((discovered, failed), (2, 0))
        self.assertEqual(
            self.archive.execute("SELECT COUNT(*) FROM sec_documents").fetchone()[0], 4
        )

        identical = article_html("Acme results")
        document_responses = {
            row[0]: (identical, 200, "")
            for row in self.archive.execute("SELECT source_url FROM sec_documents")
        }
        document_client = FakeClient(document_responses)
        fetched, failed = fetch_documents(
            self.archive, document_client, raw_dir, None, workers=4
        )
        self.assertEqual((fetched, failed), (4, 0))
        self.assertEqual(
            self.archive.execute(
                "SELECT COUNT(DISTINCT accepted_at) FROM sec_documents"
            ).fetchone()[0],
            1,
        )
        calls_before = len(document_client.calls)
        self.assertEqual(
            fetch_documents(self.archive, document_client, raw_dir, None), (0, 0)
        )
        self.assertEqual(len(document_client.calls), calls_before)

        process_archive(self.archive)
        rows = self.archive.execute(
            """
            SELECT quality_status,canonical_article_id,effective_date
            FROM articles ORDER BY id
            """
        ).fetchall()
        self.assertTrue(all(row[0] == "usable" for row in rows))
        self.assertEqual(len({row[1] for row in rows}), 1)
        self.assertTrue(all(row[2] == "2025-02-03" for row in rows))
        metadata = self.archive.execute(
            """
            SELECT COUNT(*),COUNT(byte_sha256),COUNT(raw_text_sha256),
                   COUNT(raw_path),COUNT(article_id)
            FROM sec_documents
            """
        ).fetchone()
        self.assertEqual(metadata, (4, 4, 4, 4, 4))


if __name__ == "__main__":
    unittest.main()
