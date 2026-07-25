import csv
import json
import sqlite3
import tempfile
import unittest
from datetime import date
from pathlib import Path

from news_candidate_recall import (
    FastAliasMatcher,
    MissingNameResolver,
    candidates_from_tickers,
    merge_candidates,
)
from news_events import (
    Candidate,
    candidates_for_text,
    import_alias_overrides,
    import_entities,
)


class CandidateRecallTests(unittest.TestCase):
    def registry(self, root: Path) -> sqlite3.Connection:
        entities = root / "entities.csv"
        with entities.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["ticker", "company_name", "sector"])
            writer.writerow(["AAL", "American Airlines Group Inc.", "Industrials"])
            writer.writerow(["META", "Meta Platforms Inc.", "Communication Services"])
        aliases = root / "aliases.csv"
        with aliases.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ["ticker", "alias", "alias_kind", "valid_from", "valid_to"]
            )
            writer.writerow(
                ["AAL", "American Airlines", "subsidiary", "2020-01-01", ""]
            )
            writer.writerow(
                ["META", "Facebook Inc", "sec_former_name", "2005-01-01", "2021-10-27"]
            )
        db = sqlite3.connect(":memory:")
        import_entities(db, entities)
        import_alias_overrides(db, aliases)
        return db

    def test_missing_name_resolution_is_date_bounded(self):
        with tempfile.TemporaryDirectory() as directory:
            db = self.registry(Path(directory))
            resolver = MissingNameResolver(db)
            self.assertEqual(
                resolver.resolve("American Airlines", date(2024, 1, 1))[0],
                "AAL",
            )
            self.assertEqual(
                resolver.resolve("Facebook Inc", date(2020, 1, 1))[0],
                "META",
            )
            self.assertIsNone(
                resolver.resolve("Facebook Inc", date(2024, 1, 1))
            )
            db.close()

    def test_fuzzy_resolution_requires_a_unique_multiword_match(self):
        with tempfile.TemporaryDirectory() as directory:
            db = self.registry(Path(directory))
            resolver = MissingNameResolver(db)
            result = resolver.resolve(
                "American Airlines Company", date(2024, 1, 1)
            )
            self.assertEqual(result[0], "AAL")
            self.assertIn("fuzzy", result[2])
            self.assertIsNone(resolver.resolve("American", date(2024, 1, 1)))
            db.close()

    def test_merge_preserves_old_candidate_and_combines_reasons(self):
        old = Candidate("ticker", "META", 0.92, ("ticker match",))
        new = Candidate("ticker", "META", 0.98, ("resolved name",))
        added = Candidate("ticker", "AAL", 0.99, ("provider tag",))
        merged = merge_candidates((old,), (new, added))
        by_id = {item.entity_id: item for item in merged}
        self.assertEqual(set(by_id), {"META", "AAL"})
        self.assertEqual(by_id["META"].score, 0.98)
        self.assertEqual(
            set(by_id["META"].reasons), {"ticker match", "resolved name"}
        )

    def test_fast_matcher_preserves_exact_match_semantics(self):
        with tempfile.TemporaryDirectory() as directory:
            db = self.registry(Path(directory))
            text = "META and $AAL discussed American Airlines and Facebook Inc."
            for event_date in (date(2020, 1, 1), date(2024, 1, 1)):
                expected = candidates_for_text(db, text, event_date)
                actual = FastAliasMatcher(db).candidates(text, event_date)
                self.assertEqual(actual, expected)
            db.close()

    def test_provider_tags_only_include_known_entities(self):
        candidates = candidates_from_tickers(
            ("META", "UNKNOWN"),
            {"META": "Communication Services"},
            "provider tag",
        )
        self.assertEqual(
            {(item.scope, item.entity_id) for item in candidates},
            {
                ("ticker", "META"),
                ("sector", "Communication Services"),
            },
        )


if __name__ == "__main__":
    unittest.main()
