import csv
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

import numpy as np

from chunk_articles import initialize_chunk_schema
from embed_chunks import get_or_create_config, initialize_output
from sec_features import (
    attach_vectors,
    build_metadata,
    export_daily,
    select_candidates,
    timing_features,
)
from sec_text_archive import connect_archive


class SecFeaturesTest(unittest.TestCase):
    def test_market_close_uses_new_york_timezone(self):
        self.assertEqual(timing_features("2024-02-01T20:59:00Z")[0], 0)
        self.assertEqual(timing_features("2024-02-01T21:01:00Z")[0], 1)
        self.assertEqual(timing_features("2024-07-01T20:01:00Z")[0], 1)

    def test_metadata_vectors_and_selection_form_one_consistent_event(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive_path = root / "archive.sqlite3"
            embeddings_path = root / "embeddings.sqlite3"
            features_path = root / "features.sqlite3"
            selection_path = root / "selection.csv"
            calendar_path = root / "calendar.csv"
            daily_path = root / "daily.csv"
            version = "paragraph-v1-500-600-75"

            with connect_archive(archive_path) as db:
                initialize_chunk_schema(db)
                db.execute(
                    "INSERT INTO chunking_configs VALUES(?,?,?,?,?,?)",
                    (version, "test", 500, 600, 75, "{}"),
                )
                db.execute(
                    """
                    INSERT INTO sec_filing_queue(
                        accession,cik,filing_date,accepted_at,
                        available_at_quality,form,primary_document,
                        source_metadata_retrieved_at,status,index_url
                    ) VALUES('a',1,'2021-01-04','2021-01-04T21:05:00Z',
                             'exact','8-K','main.htm','2021-01-04T22:00:00Z',
                             'complete','https://example/index')
                    """
                )
                db.execute(
                    "INSERT INTO sec_filing_tickers VALUES('a','TEST','Test Inc.','Industrials')"
                )
                text = (
                    "Item 2.02 Results of Operations. Revenue increased 20%. "
                    "The company may face litigation risk."
                )
                article_id = db.execute(
                    """
                    INSERT INTO articles(
                        source_url,domain,title,article_text,article_text_raw,
                        article_text_clean,status,quality_status,word_count,
                        cleaning_version
                    ) VALUES('https://example/ex99','sec.gov','Results',?,?,?,
                             'ok','usable',15,'test-v1')
                    """,
                    (text, text, text),
                ).lastrowid
                db.execute(
                    """
                    INSERT INTO sec_documents(
                        accession,cik,form,accepted_at,document_name,description,
                        selection_reason,source_url,is_primary,status,article_id
                    ) VALUES('a',1,'8-K','2021-01-04T21:05:00Z','ex99.htm',
                             'Press release','exhibit_99','https://example/ex99',
                             0,'ok',?)
                    """,
                    (article_id,),
                )
                db.execute(
                    """
                    INSERT INTO article_chunks(
                        article_id,chunking_version,cleaning_version,source_hash,
                        chunk_index,body_text,embedding_text,token_count,
                        paragraph_start,paragraph_end,chunk_hash
                    ) VALUES(?,?, 'test-v1','source',0,?,?,10,0,0,'chunk')
                    """,
                    (article_id, version, text, text),
                )
                chunk_id = db.execute(
                    "SELECT id FROM article_chunks"
                ).fetchone()[0]
                db.commit()

            with sqlite3.connect(embeddings_path) as db:
                initialize_output(db)
                config_id = get_or_create_config(
                    db, "test-model", "revision", 2, version
                )
                vector = np.asarray([0.6, 0.8], dtype="<f2").tobytes()
                db.execute(
                    "INSERT INTO chunk_embeddings(config_id,chunk_id,article_id,chunk_hash,vector) VALUES(?,?,?,?,?)",
                    (config_id, chunk_id, article_id, "chunk", vector),
                )
                db.commit()

            self.assertEqual(build_metadata(archive_path, features_path)["events"], 1)
            self.assertEqual(
                attach_vectors(archive_path, embeddings_path, features_path)["chunks"],
                1,
            )
            result = select_candidates(
                features_path, selection_path, 1, 0.6, 0.2
            )
            self.assertEqual(result["selected"], 1)
            with sqlite3.connect(features_path) as db:
                row = db.execute(
                    """
                    SELECT item_codes_json,exhibit99_count,after_market_close,
                           dimension,vector FROM sec_event_features
                    """
                ).fetchone()
            self.assertEqual(json.loads(row[0]), ["2.02"])
            self.assertEqual(row[1:4], (1, 1, 2))
            self.assertTrue(
                np.allclose(
                    np.frombuffer(row[4], dtype="<f2"),
                    [0.6, 0.8],
                    atol=1e-3,
                )
            )
            with selection_path.open() as handle:
                selected = list(csv.DictReader(handle))
            self.assertEqual(selected[0]["event_id"], "sec:a")
            self.assertEqual(selected[0]["ticker"], "TEST")
            calendar_path.write_text(
                "date\n2021-01-04\n2021-01-05\n2021-01-06\n"
            )
            self.assertEqual(
                export_daily(features_path, calendar_path, daily_path),
                (1, 0),
            )
            with daily_path.open() as handle:
                daily = list(csv.DictReader(handle))
            self.assertEqual(daily[0]["trade_date"], "2021-01-06")
            self.assertEqual(daily[0]["sec_item_2_02_count"], "1")


if __name__ == "__main__":
    unittest.main()
