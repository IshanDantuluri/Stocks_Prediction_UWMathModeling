import sqlite3
import tempfile
import unittest
from pathlib import Path

from news_archive import (
    CLEANING_VERSION,
    clean_article,
    clean_corpus,
    extract_article,
    initialize,
    update_effective_dates,
)


class ExtractionTests(unittest.TestCase):
    def test_extracts_article_and_metadata_but_not_page_chrome(self):
        html = b"""
        <html lang="en"><head>
          <meta property="og:title" content="A useful headline | Example News">
          <meta name="author" content="Jane Reporter">
          <meta property="article:published_time" content="2025-02-03T10:00:00Z">
        </head><body>
          <nav><p>This navigation paragraph is long enough to otherwise qualify.</p></nav>
          <article>
            <p>By Jane Reporter</p>
            <p>The opening paragraph contains the principal facts of the reported event.</p>
            <p>The second paragraph supplies enough context to make the story meaningful.</p>
            <div class="newsletter"><p>Sign up for our newsletter and daily updates.</p></div>
          </article>
          <footer><p>This footer paragraph should never enter the extracted article.</p></footer>
        </body></html>
        """
        item = extract_article(html)
        self.assertEqual(item["language"], "en")
        self.assertEqual(item["author"], "Jane Reporter")
        self.assertIn("principal facts", item["text"])
        self.assertNotIn("navigation", item["text"])
        self.assertNotIn("newsletter", item["text"])

    def test_cleaning_is_conservative_and_versionable(self):
        raw = "\n\n".join([
            "By Jane Reporter",
            "Officials announced a major policy change after a lengthy public review.",
            "The policy will affect several industries and take effect later this year.",
            "Analysts said the consequences would depend on implementation details.",
            "Stay in the know on news that matters to you with our newsletter.",
        ])
        item = clean_article(raw, "Policy changes announced | Example News")
        self.assertEqual(item["title"], "Policy changes announced")
        self.assertNotIn("Jane Reporter", item["text"])
        self.assertNotIn("Stay in the know", item["text"])
        self.assertIn("implementation details", item["text"])

    def test_removes_entire_promotional_tail(self):
        raw = "\n\n".join([
            "The first substantial paragraph reports the event and identifies the people involved.",
            "A second substantial paragraph provides important context and relevant background information.",
            "The final reporting paragraph explains what officials expect to happen next.",
            "Be the first to know when news breaks.",
            "Today's top stories curated by our news team.",
            "Your digital replica of Today's Paper. Ready to read from 5am!",
            "Test your skills with interactive crosswords, sudoku and trivia.",
        ])
        item = clean_article(raw, "A real article | Example")
        self.assertIn("officials expect", item["text"])
        self.assertNotIn("top stories", item["text"])
        self.assertNotIn("crosswords", item["text"])
        self.assertIn("promotional_tail_removed", item["reasons"])

    def test_collapses_cumulative_and_exact_paragraphs(self):
        short = "Officials announced the policy after meeting regional representatives and industry groups."
        long = short + " The implementation process will begin next month after a public consultation."
        raw = "\n\n".join([short, long, long, "A separate final paragraph contains additional reporting."])
        item = clean_article(raw, "Policy announcement")
        self.assertEqual(item["text"].count("Officials announced"), 1)
        self.assertIn("implementation process", item["text"])
        self.assertIn("duplicate_fragments_removed", item["reasons"])


class CorpusTests(unittest.TestCase):
    def test_effective_date_never_moves_article_earlier(self):
        with sqlite3.connect(":memory:") as db:
            initialize(db)
            db.execute(
                """INSERT INTO articles(
                     id,source_url,status,published_at,canonical_article_id
                   ) VALUES (1,'https://example.com/later','ok',
                             '2020-01-05T12:00:00Z',1)"""
            )
            db.execute(
                """INSERT INTO articles(
                     id,source_url,status,published_at,canonical_article_id
                   ) VALUES (2,'https://example.com/earlier','ok',
                             '2020-01-01T12:00:00Z',2)"""
            )
            db.execute(
                "INSERT INTO events(id,date,article_id) VALUES (1,'2020-01-03',1)"
            )
            db.execute(
                "INSERT INTO events(id,date,article_id) VALUES (2,'2020-01-03',2)"
            )
            stats = update_effective_dates(db)
            rows = db.execute(
                """SELECT effective_date,effective_date_source FROM articles
                   ORDER BY id"""
            ).fetchall()
            self.assertEqual(
                rows,
                [
                    ("2020-01-05", "published_at_later"),
                    ("2020-01-03", "event_date_published_not_later"),
                ],
            )
            self.assertEqual(stats["moved_later"], 1)

    def test_schema_upgrade_and_duplicate_linking(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.sqlite3"
            db = sqlite3.connect(path)
            initialize(db)
            text = "\n\n".join([
                "This is a substantial opening paragraph about an international event, its participants, and the communities directly affected by it.",
                "Officials provided additional details about the event, the decisions already taken, and the expected consequences over the coming months.",
                "Independent analysts discussed the historical background, comparable earlier developments, and the likely next steps for everyone involved.",
                "Several organizations said they would continue monitoring conditions and publish additional findings as reliable information becomes available.",
                "Residents described how the announcement changed their immediate plans while local agencies prepared detailed guidance for the public.",
                "Researchers cautioned that the longer-term effects remain uncertain and that conclusions should be revisited when more evidence is available.",
                "The responsible department plans to issue another formal update after consulting regional officials and representatives from affected industries.",
            ])
            for url in ("https://a.example/story", "https://b.example/reprint"):
                db.execute(
                    "INSERT INTO articles(source_url,domain,title,article_text_raw,status) VALUES (?,?,?,?,?)",
                    (url, url.split('/')[2], "International event reported", text, "ok"),
                )
            db.commit()
            clean_corpus(db)
            rows = db.execute(
                "SELECT id,cleaning_version,quality_status,canonical_article_id FROM articles ORDER BY id"
            ).fetchall()
            self.assertEqual(rows[0][1], CLEANING_VERSION)
            self.assertEqual(rows[0][2], "usable")
            self.assertEqual(rows[0][3], rows[0][0])
            self.assertEqual(rows[1][3], rows[0][0])
            db.close()

    def test_rejects_same_fallback_page_for_many_domain_urls(self):
        with tempfile.TemporaryDirectory() as directory:
            db = sqlite3.connect(Path(directory) / "test.sqlite3")
            initialize(db)
            text = "\n\n".join(
                f"Homepage news item number {i} contains unrelated current headlines and descriptive filler."
                for i in range(20)
            )
            for i in range(3):
                db.execute(
                    "INSERT INTO articles(source_url,domain,title,article_text_raw,status) VALUES (?,?,?,?,?)",
                    (f"https://example.com/missing/{i}", "example.com", "Latest News", text, "ok"),
                )
            db.commit()
            clean_corpus(db)
            statuses = db.execute("SELECT DISTINCT quality_status FROM articles").fetchall()
            reasons = db.execute("SELECT quality_reasons FROM articles LIMIT 1").fetchone()[0]
            self.assertEqual(statuses, [("rejected",)])
            self.assertIn("repeated_domain_fallback", reasons)
            db.close()


if __name__ == "__main__":
    unittest.main()
