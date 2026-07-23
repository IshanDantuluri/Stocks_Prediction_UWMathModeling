import sqlite3
import unittest

from benchmark_embedder import evenly_spaced, load_sample


class BenchmarkEmbedderTests(unittest.TestCase):
    def test_even_sample_spans_population(self):
        self.assertEqual(evenly_spaced(list(range(10)), 3), [0, 3, 6])
        self.assertEqual(evenly_spaced([1, 2], 5), [1, 2])

    def test_load_sample_only_uses_canonical_articles(self):
        db = sqlite3.connect(":memory:")
        db.executescript(
            """
            CREATE TABLE articles (
                id INTEGER PRIMARY KEY,
                quality_status TEXT,
                canonical_article_id INTEGER
            );
            CREATE TABLE article_chunks (
                article_id INTEGER,
                chunking_version TEXT,
                chunk_index INTEGER,
                embedding_text TEXT,
                token_count INTEGER
            );
            INSERT INTO articles VALUES (1, 'usable', 1), (2, 'usable', 1),
                                        (3, 'questionable', 3), (4, 'usable', 4);
            INSERT INTO article_chunks VALUES
                (1, 'v1', 0, 'one-a', 10), (1, 'v1', 1, 'one-b', 11),
                (2, 'v1', 0, 'duplicate', 12), (3, 'v1', 0, 'bad', 13),
                (4, 'v1', 0, 'four', 14), (4, 'old', 0, 'old-four', 15);
            """
        )
        rows, population, selected = load_sample(db, "v1", 500)
        self.assertEqual(population, 2)
        self.assertEqual(selected, 2)
        self.assertEqual(rows, [(1, "one-a", 10), (1, "one-b", 11), (4, "four", 14)])


if __name__ == "__main__":
    unittest.main()
