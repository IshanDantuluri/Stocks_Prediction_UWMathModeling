import sqlite3
import unittest

from embed_chunks import canonical_chunk_query, get_or_create_config, initialize_output


class EmbedChunksTests(unittest.TestCase):
    def test_config_is_idempotent_and_versioned(self):
        db = sqlite3.connect(":memory:")
        initialize_output(db)
        first = get_or_create_config(db, "model", "revision-a", 1024, "chunks-v1")
        second = get_or_create_config(db, "model", "revision-a", 1024, "chunks-v1")
        different = get_or_create_config(db, "model", "revision-b", 1024, "chunks-v1")
        self.assertEqual(first, second)
        self.assertNotEqual(first, different)

    def test_query_selects_canonical_usable_chunks(self):
        db = sqlite3.connect(":memory:")
        db.executescript(
            """
            CREATE TABLE articles(id INTEGER PRIMARY KEY, quality_status TEXT,
                                  canonical_article_id INTEGER);
            CREATE TABLE article_chunks(id INTEGER PRIMARY KEY, article_id INTEGER,
                chunking_version TEXT, chunk_hash TEXT, embedding_text TEXT, token_count INTEGER);
            INSERT INTO articles VALUES (1,'usable',1),(2,'usable',1),(3,'rejected',3);
            INSERT INTO article_chunks VALUES
                (10,1,'v1','a','text a',20),(11,2,'v1','b','text b',20),
                (12,3,'v1','c','text c',20),(13,1,'old','d','text d',20);
            """
        )
        sql, tail = canonical_chunk_query()
        rows = db.execute(sql, ("v1", *tail)).fetchall()
        self.assertEqual(rows, [(10, 1, "a", "text a", 20)])


if __name__ == "__main__":
    unittest.main()
