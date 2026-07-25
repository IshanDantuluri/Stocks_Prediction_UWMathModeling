import csv
import gzip
import json
import sqlite3
import tempfile
import unittest
from datetime import date
from pathlib import Path

from entity_knowledge import curated_aliases, sec_aliases, write_csv
from news_events import candidates_for_text, import_alias_overrides, import_entities


class EntityKnowledgeTests(unittest.TestCase):
    def make_point_in_time_db(self, root: Path) -> Path:
        database = root / "pit.sqlite3"
        raw = root / "raw.json.gz"
        with gzip.open(raw, "wt", encoding="utf-8") as handle:
            json.dump(
                {
                    "name": "Meta Platforms, Inc.",
                    "formerNames": [
                        {
                            "name": "Facebook Inc",
                            "from": "2005-05-06T04:00:00.000Z",
                            "to": "2021-10-27T04:00:00.000Z",
                        }
                    ],
                },
                handle,
            )
        with sqlite3.connect(database) as db:
            db.execute("CREATE TABLE entities(ticker,cik)")
            db.execute(
                """CREATE TABLE source_objects(
                     source,object_key,raw_path,retrieved_at)"""
            )
            db.execute("INSERT INTO entities VALUES ('META',1326801)")
            db.execute(
                "INSERT INTO source_objects VALUES (?,?,?,?)",
                (
                    "sec",
                    "submissions/CIK0001326801",
                    raw.name,
                    "2026-07-23T12:00:00+00:00",
                ),
            )
        return database

    def test_sec_names_retain_historical_validity(self):
        with tempfile.TemporaryDirectory() as directory:
            records = sec_aliases(self.make_point_in_time_db(Path(directory)))
            former = next(item for item in records if item.alias == "Facebook Inc")
            current = next(
                item for item in records if item.alias == "Meta Platforms, Inc."
            )
            self.assertEqual(former.valid_from, "2005-05-06")
            self.assertEqual(former.valid_to, "2021-10-27")
            self.assertEqual(current.valid_from, "2021-10-28")

    def test_prospective_curated_alias_requires_start_date(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "curated.csv"
            with source.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["ticker", "alias", "alias_kind", "valid_from"])
                writer.writerow(["META", "Quest", "product", ""])
            with self.assertRaisesRegex(ValueError, "requires valid_from"):
                curated_aliases(source)

    def test_import_preserves_kind_and_candidate_dates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entities = root / "entities.csv"
            with entities.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["ticker", "company_name"])
                writer.writerow(["META", "Meta Platforms"])
            alias_csv = root / "aliases.csv"
            write_csv(alias_csv, sec_aliases(self.make_point_in_time_db(root)))
            with sqlite3.connect(":memory:") as db:
                import_entities(db, entities)
                imported, skipped = import_alias_overrides(db, alias_csv)
                self.assertEqual((imported, skipped), (2, 0))
                kind = db.execute(
                    "SELECT alias_kind FROM entity_aliases WHERE alias='Facebook Inc'"
                ).fetchone()[0]
                self.assertEqual(kind, "sec_former_name")
                import_entities(db, entities)
                self.assertEqual(
                    db.execute(
                        """SELECT COUNT(*) FROM entity_aliases
                           WHERE alias_kind='sec_former_name'"""
                    ).fetchone()[0],
                    1,
                )
                old = candidates_for_text(
                    db, "Facebook Inc announced a deal", date(2020, 1, 1)
                )
                new = candidates_for_text(
                    db, "Facebook Inc announced a deal", date(2024, 1, 1)
                )
                self.assertIn("META", [item.entity_id for item in old])
                self.assertNotIn("META", [item.entity_id for item in new])


if __name__ == "__main__":
    unittest.main()
