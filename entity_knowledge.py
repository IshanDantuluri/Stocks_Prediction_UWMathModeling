#!/usr/bin/env python3
"""Build date-safe entity aliases from cached SEC and curated knowledge.

The output is compatible with ``news_events.py import-aliases``.  SEC former
names retain their reported validity intervals.  Present-day knowledge such as
brands, products, subsidiaries, and executives is accepted only with an
explicit ``valid_from`` date; this prevents current knowledge from leaking
backwards into historical event linking.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import sqlite3
from dataclasses import dataclass, fields
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable

from news_events import normalize_alias


DEFAULT_POINT_IN_TIME_DB = Path("point_in_time_data.sqlite3")
DEFAULT_OUTPUT = Path("entity_alias_enrichment.csv")
PROSPECTIVE_KINDS = frozenset(
    {"brand", "product", "subsidiary", "executive", "prospective"}
)


@dataclass(frozen=True)
class AliasRecord:
    ticker: str
    alias: str
    alias_kind: str
    valid_from: str
    valid_to: str
    source: str
    confidence: str
    retrieved_at: str


def date_prefix(value: str | None) -> str:
    if not value:
        return ""
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).date().isoformat()
    except ValueError:
        return value[:10]


def next_day(value: str) -> str:
    return (date.fromisoformat(value) + timedelta(days=1)).isoformat()


def resolve_raw_path(database: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else database.parent / path


def sec_aliases(database: Path) -> list[AliasRecord]:
    """Extract legal/former names from cached main SEC submission objects."""
    uri = f"file:{database.resolve()}?mode=ro"
    records: list[AliasRecord] = []
    with sqlite3.connect(uri, uri=True) as db:
        mappings = db.execute(
            """SELECT e.ticker,e.cik,s.raw_path,s.retrieved_at
               FROM entities e JOIN source_objects s
                 ON s.source='sec'
                AND s.object_key=printf('submissions/CIK%010d',e.cik)
               WHERE e.cik IS NOT NULL
               ORDER BY e.ticker"""
        ).fetchall()
    for ticker, _cik, raw_path, retrieved_at in mappings:
        path = resolve_raw_path(database, raw_path)
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
        former_names = payload.get("formerNames") or []
        latest_former_to = ""
        for item in former_names:
            alias = str(item.get("name") or "").strip()
            valid_from = date_prefix(item.get("from"))
            valid_to = date_prefix(item.get("to"))
            if not alias:
                continue
            latest_former_to = max(latest_former_to, valid_to)
            records.append(
                AliasRecord(
                    ticker=ticker,
                    alias=alias,
                    alias_kind="sec_former_name",
                    valid_from=valid_from,
                    valid_to=valid_to,
                    source="sec_submissions",
                    confidence="1.0",
                    retrieved_at=retrieved_at,
                )
            )
        legal_name = str(payload.get("name") or "").strip()
        if legal_name:
            # SEC does not report a start date for the current name.  The day
            # after the latest former-name end is safe when one is available;
            # otherwise retrieval date is conservative and prospective-only.
            valid_from = (
                next_day(latest_former_to)
                if latest_former_to
                else date_prefix(retrieved_at)
            )
            records.append(
                AliasRecord(
                    ticker=ticker,
                    alias=legal_name,
                    alias_kind="sec_legal_name",
                    valid_from=valid_from,
                    valid_to="",
                    source="sec_submissions",
                    confidence="1.0",
                    retrieved_at=retrieved_at,
                )
            )
    return records


def curated_aliases(path: Path) -> list[AliasRecord]:
    """Load hand-reviewed aliases while enforcing prospective time bounds."""
    records: list[AliasRecord] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"ticker", "alias", "alias_kind", "valid_from"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"curated alias CSV missing columns: {sorted(missing)}")
        for line, row in enumerate(reader, 2):
            ticker = str(row.get("ticker") or "").strip().upper()
            alias = str(row.get("alias") or "").strip()
            kind = str(row.get("alias_kind") or "").strip().casefold()
            valid_from = str(row.get("valid_from") or "").strip()
            if not ticker or not alias or not kind:
                raise ValueError(f"incomplete curated alias on line {line}")
            if kind in PROSPECTIVE_KINDS and not valid_from:
                raise ValueError(
                    f"{kind} alias on line {line} requires valid_from"
                )
            records.append(
                AliasRecord(
                    ticker=ticker,
                    alias=alias,
                    alias_kind=kind,
                    valid_from=valid_from,
                    valid_to=str(row.get("valid_to") or "").strip(),
                    source=str(row.get("source") or "curated").strip(),
                    confidence=str(row.get("confidence") or "1.0").strip(),
                    retrieved_at=str(row.get("retrieved_at") or "").strip(),
                )
            )
    return records


def deduplicate(records: Iterable[AliasRecord]) -> list[AliasRecord]:
    selected: dict[tuple[str, str, str, str, str], AliasRecord] = {}
    for record in records:
        key = (
            record.ticker,
            normalize_alias(record.alias),
            record.alias_kind,
            record.valid_from,
            record.valid_to,
        )
        selected[key] = record
    ordered = sorted(
        selected.values(),
        key=lambda item: (
            item.ticker,
            item.valid_from,
            item.valid_to,
            item.alias_kind,
            item.alias.casefold(),
        ),
    )
    # The event database historically keyed aliases without their validity
    # interval. Preserve repeated legal-name intervals by namespacing only the
    # later collisions; candidate scoring treats the prefix identically.
    seen: set[tuple[str, str, str]] = set()
    result = []
    for record in ordered:
        key = (record.ticker, normalize_alias(record.alias), record.alias_kind)
        if key in seen:
            record = AliasRecord(
                **{
                    **record.__dict__,
                    "alias_kind": f"{record.alias_kind}:{record.valid_from or 'undated'}",
                }
            )
        seen.add(key)
        result.append(record)
    return result


def write_csv(path: Path, records: Iterable[AliasRecord]) -> int:
    rows = deduplicate(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[field.name for field in fields(AliasRecord)])
        writer.writeheader()
        writer.writerows(
            {field.name: getattr(record, field.name) for field in fields(AliasRecord)}
            for record in rows
        )
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--point-in-time-db", type=Path, default=DEFAULT_POINT_IN_TIME_DB)
    parser.add_argument("--curated", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    records = sec_aliases(args.point_in_time_db)
    sec_count = len(records)
    for path in args.curated:
        records.extend(curated_aliases(path))
    total = write_csv(args.output, records)
    print(
        f"Wrote {total:,} date-safe aliases to {args.output} "
        f"({sec_count:,} from cached SEC submissions)",
        flush=True,
    )


if __name__ == "__main__":
    main()
