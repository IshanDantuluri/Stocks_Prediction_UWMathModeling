import sqlite3
import tempfile
import unittest
from pathlib import Path

from chunk_articles import make_chunks, sync_chunks
from news_archive import initialize


class WordTokenizer:
    def __init__(self):
        self.values = {}
        self.reverse = {}

    def encode(self, text, add_special_tokens=False):
        result = []
        for word in text.split():
            if word not in self.values:
                value = len(self.values) + 1
                self.values[word] = value
                self.reverse[value] = word
            result.append(self.values[word])
        return result

    def decode(self, token_ids, skip_special_tokens=True):
        return " ".join(self.reverse[token] for token in token_ids)


class ChunkingTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = WordTokenizer()

    def test_short_article_remains_one_chunk(self):
        text = "First complete paragraph has useful details.\n\nSecond paragraph adds context."
        chunks = make_chunks("Headline", text, self.tokenizer, 50, 60, 10)
        self.assertEqual(len(chunks), 1)
        self.assertTrue(chunks[0].embedding_text.startswith("Title: Headline"))
        self.assertIn("Second paragraph", chunks[0].body_text)

    def test_chunks_respect_maximum_and_overlap(self):
        paragraphs = [" ".join(f"p{i}word{j}" for j in range(24)) + "." for i in range(8)]
        chunks = make_chunks("Test headline", "\n\n".join(paragraphs), self.tokenizer, 70, 80, 25)
        self.assertGreater(len(chunks), 2)
        self.assertTrue(all(chunk.token_count <= 80 for chunk in chunks))
        self.assertLess(chunks[1].paragraph_start, chunks[0].paragraph_end + 1)

    def test_oversized_paragraph_splits_at_sentences(self):
        paragraph = " ".join((f"Sentence {i} contains several useful words." for i in range(30)))
        chunks = make_chunks("Headline", paragraph, self.tokenizer, 50, 60, 10)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk.token_count <= 60 for chunk in chunks))

    def test_merges_tiny_final_fragment_when_it_fits(self):
        paragraphs = [" ".join(f"word{i}_{j}" for j in range(20)) for i in range(4)]
        paragraphs.append("tiny footer")
        chunks = make_chunks(
            "Headline", "\n\n".join(paragraphs), self.tokenizer,
            target_tokens=55, max_tokens=70, overlap_tokens=10, minimum_tokens=15,
        )
        self.assertGreater(len(chunks), 1)
        self.assertGreaterEqual(chunks[-1].token_count, 15)
        self.assertTrue(all(chunk.token_count <= 70 for chunk in chunks))

    def test_database_sync_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            db = sqlite3.connect(Path(directory) / "test.sqlite3")
            initialize(db)
            text = "\n\n".join(" ".join(f"word{i}_{j}" for j in range(30)) for i in range(5))
            db.execute(
                """INSERT INTO articles(source_url,domain,title,article_text_clean,status,
                   cleaning_version,quality_status) VALUES (?,?,?,?,?,?,?)""",
                ("https://example.com/a", "example.com", "Headline", text,
                 "ok", "test-clean", "usable"),
            )
            db.commit()
            first = sync_chunks(db, self.tokenizer, "word-test", 70, 80, 20)
            second = sync_chunks(db, self.tokenizer, "word-test", 70, 80, 20)
            self.assertEqual(first[0], 1)
            self.assertEqual(second[0], 0)
            self.assertEqual(second[1], 1)
            self.assertEqual(first[2], second[2])
            db.close()


if __name__ == "__main__":
    unittest.main()
