import sqlite3
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from deepseek_reasoner import load_linked_work
from sec_events import (
    DIRECT_LINK_MODEL,
    DIRECT_LINK_PROMPT,
    build,
)
from sec_text_archive import connect_archive


class SecEventsTest(unittest.TestCase):
    def test_groups_accession_and_creates_reasoner_ready_issuer_link(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive_path = root / "archive.sqlite3"
            output_path = root / "events.sqlite3"
            with connect_archive(archive_path) as archive:
                archive.execute(
                    """
                    INSERT INTO sec_filing_queue(
                        accession,cik,filing_date,accepted_at,
                        available_at_quality,form,primary_document,
                        source_metadata_retrieved_at,status,index_url
                    ) VALUES(
                        '0000000001-24-000001',1,'2024-02-01',
                        '2024-02-01T21:05:00Z','exact','8-K','main.htm',
                        '2024-02-01T22:00:00Z','complete','https://example/index'
                    )
                    """
                )
                archive.execute(
                    """
                    INSERT INTO sec_filing_tickers(
                        accession,ticker,company_name,sector
                    ) VALUES('0000000001-24-000001','TEST','Test Holdings, Inc.',
                             'Industrials')
                    """
                )
                article_ids = []
                for number, (name, reason, title, text) in enumerate(
                    (
                        (
                            "ex99.htm",
                            "exhibit_99",
                            "Test announces quarterly results",
                            "Revenue increased and management updated guidance.",
                        ),
                        (
                            "main.htm",
                            "primary_document",
                            "Form 8-K",
                            "The company furnished an earnings release.",
                        ),
                    ),
                    1,
                ):
                    url = f"https://example/{name}"
                    cursor = archive.execute(
                        """
                        INSERT INTO articles(
                            source_url,domain,title,article_text,article_text_raw,
                            article_text_clean,status,quality_status,published_at,
                            effective_date,cleaning_version
                        ) VALUES(?,?,?,?,?,?,'ok','usable',
                                 '2024-02-01T21:05:00Z','2024-02-01','test-v1')
                        """,
                        (url, "sec.gov", title, text, text, text),
                    )
                    article_id = int(cursor.lastrowid)
                    article_ids.append(article_id)
                    archive.execute(
                        """
                        INSERT INTO sec_documents(
                            accession,cik,form,accepted_at,document_name,
                            selection_reason,source_url,is_primary,status,article_id
                        ) VALUES(
                            '0000000001-24-000001',1,'8-K',
                            '2024-02-01T21:05:00Z',?,?,?,?,'ok',?
                        )
                        """,
                        (
                            name,
                            reason,
                            url,
                            int(reason == "primary_document"),
                            article_id,
                        ),
                    )
                archive.commit()

            result = build(
                Namespace(
                    archive=archive_path,
                    embeddings=root / "missing-embeddings.sqlite3",
                    output=output_path,
                    max_event_chars=16_000,
                    progress_every=100,
                    limit=None,
                    dry_run=False,
                )
            )
            self.assertEqual(result["events"], 1)
            self.assertEqual(result["documents"], 2)
            self.assertEqual(result["issuer_links"], 1)

            config_id, work = load_linked_work(
                output_path,
                archive_path,
                linker_model=DIRECT_LINK_MODEL,
                linker_prompt=DIRECT_LINK_PROMPT,
                scopes=("ticker",),
            )
            self.assertEqual(config_id, result["config_id"])
            self.assertEqual(len(work), 1)
            self.assertEqual(work[0].entity.entity_id, "TEST")
            self.assertEqual(work[0].event.article_ids, tuple(article_ids))
            self.assertIn(
                "Revenue increased",
                work[0].event.summary,
            )
            self.assertIn(
                "issuer ticker(s) TEST",
                work[0].event.summary,
            )


if __name__ == "__main__":
    unittest.main()
