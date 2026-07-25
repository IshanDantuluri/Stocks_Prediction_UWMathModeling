#!/usr/bin/env python3
"""Build deterministic SEC event features, vectors, and a ranked LLM sample.

No realized market outcome is read by this module.  Selection uses only filing
metadata, contemporaneous text, issuer coverage, and embedding novelty relative
to an issuer's earlier filing.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import math
import re
import sqlite3
import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np


DEFAULT_ARCHIVE = Path("sec_text_archive.sqlite3")
DEFAULT_EMBEDDINGS = Path("sec_embeddings_a100.sqlite3")
DEFAULT_OUTPUT = Path("sec_features.sqlite3")
DEFAULT_SELECTION = Path("sec_deepseek_candidates.csv")
ITEM_RE = re.compile(r"\bitem\s+(\d{1,2}\.\d{2})\b", re.IGNORECASE)
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'-]*")
NUMBER_RE = re.compile(r"(?<![A-Za-z])[-+]?\d[\d,]*(?:\.\d+)?")
CURRENCY_RE = re.compile(r"[$€£¥]\s*\d")
PERCENT_RE = re.compile(r"\d(?:\.\d+)?\s*%")
MARKET_TZ = ZoneInfo("America/New_York")

LEXICONS = {
    "positive": {
        "achieve", "benefit", "exceed", "gain", "growth", "improve", "increase",
        "opportunity", "profit", "record", "strong", "success",
    },
    "negative": {
        "adverse", "decline", "decrease", "deteriorate", "fail", "impairment",
        "loss", "negative", "risk", "weak", "worse",
    },
    "uncertainty": {
        "approximately", "contingent", "could", "depend", "may", "might",
        "possible", "uncertain", "uncertainty", "unknown",
    },
    "litigation": {
        "action", "claim", "court", "investigation", "lawsuit", "legal",
        "litigation", "plaintiff", "proceeding", "settlement",
    },
    "constraining": {
        "covenant", "limit", "must", "obligation", "prohibit", "require",
        "restrict", "restriction", "shall",
    },
}

ITEM_WEIGHTS = {
    "1.01": 3.0,  # material agreement
    "1.03": 5.0,  # bankruptcy
    "2.01": 5.0,  # acquisition/disposition
    "2.02": 5.0,  # results of operations
    "2.03": 3.5,  # obligations
    "2.04": 3.5,  # triggering events
    "3.01": 5.0,  # delisting
    "4.01": 4.0,  # auditor change
    "4.02": 5.0,  # non-reliance/restatement
    "5.02": 3.0,  # management change
    "7.01": 1.5,  # Regulation FD
    "8.01": 1.0,  # other events
}


@dataclass
class EventAccumulator:
    accession: str
    cik: int
    filing_date: str
    accepted_at: str
    form: str
    document_count: int = 0
    exhibit99_count: int = 0
    primary_count: int = 0
    press_release_count: int = 0
    char_count: int = 0
    word_count: int = 0
    number_count: int = 0
    currency_count: int = 0
    percent_count: int = 0
    positive_count: int = 0
    negative_count: int = 0
    uncertainty_count: int = 0
    litigation_count: int = 0
    constraining_count: int = 0
    item_codes: set[str] | None = None

    def __post_init__(self) -> None:
        self.item_codes = set()


def initialize(db: sqlite3.Connection) -> None:
    db.executescript(
        """
        PRAGMA foreign_keys=ON;
        CREATE TABLE IF NOT EXISTS feature_configs (
            id INTEGER PRIMARY KEY,
            config_hash TEXT NOT NULL UNIQUE,
            config_json TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS sec_event_features (
            event_id TEXT PRIMARY KEY,
            accession TEXT NOT NULL UNIQUE,
            cik INTEGER NOT NULL,
            filing_date TEXT NOT NULL,
            accepted_at TEXT NOT NULL,
            form TEXT NOT NULL,
            item_codes_json TEXT NOT NULL,
            importance_score REAL NOT NULL,
            after_market_close INTEGER NOT NULL,
            accepted_hour_et REAL NOT NULL,
            document_count INTEGER NOT NULL,
            exhibit99_count INTEGER NOT NULL,
            primary_count INTEGER NOT NULL,
            press_release_count INTEGER NOT NULL,
            char_count INTEGER NOT NULL,
            word_count INTEGER NOT NULL,
            number_count INTEGER NOT NULL,
            currency_count INTEGER NOT NULL,
            percent_count INTEGER NOT NULL,
            positive_count INTEGER NOT NULL,
            negative_count INTEGER NOT NULL,
            uncertainty_count INTEGER NOT NULL,
            litigation_count INTEGER NOT NULL,
            constraining_count INTEGER NOT NULL,
            embedding_config_id INTEGER,
            dimension INTEGER,
            vector BLOB,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS sec_event_entities (
            event_id TEXT NOT NULL REFERENCES sec_event_features(event_id),
            ticker TEXT NOT NULL,
            company_name TEXT NOT NULL,
            sector TEXT,
            days_since_previous REAL,
            embedding_novelty REAL,
            PRIMARY KEY(event_id,ticker)
        );
        CREATE INDEX IF NOT EXISTS sec_event_date_idx
            ON sec_event_features(accepted_at,event_id);
        CREATE INDEX IF NOT EXISTS sec_entity_ticker_idx
            ON sec_event_entities(ticker,event_id);
        """
    )


def parse_accepted_at(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(f"SEC accepted_at lacks timezone: {value}")
    return parsed


def timing_features(accepted_at: str) -> tuple[int, float]:
    local = parse_accepted_at(accepted_at).astimezone(MARKET_TZ)
    hour = local.hour + local.minute / 60 + local.second / 3600
    return int(hour >= 16.0), hour


def lexical_counts(text: str) -> dict[str, int]:
    words = [match.group(0).lower() for match in TOKEN_RE.finditer(text)]
    frequencies = Counter(words)
    return {
        name: sum(frequencies[word] for word in lexicon)
        for name, lexicon in LEXICONS.items()
    }


def importance_score(event: EventAccumulator) -> float:
    item_score = max(
        (ITEM_WEIGHTS.get(item, 0.5) for item in event.item_codes or ()),
        default=0.0,
    )
    score = item_score
    score += 3.0 if event.exhibit99_count else 0.0
    score += 2.0 if event.press_release_count else 0.0
    score += 0.75 if event.form.upper().startswith("6-K") else 0.0
    score += min(math.log1p(event.word_count) / 10, 1.0)
    return round(score, 6)


def load_event_accumulators(archive: sqlite3.Connection) -> dict[str, EventAccumulator]:
    events: dict[str, EventAccumulator] = {}
    rows = archive.execute(
        """
        SELECT q.accession,q.cik,q.filing_date,q.accepted_at,q.form,
               d.selection_reason,COALESCE(d.description,''),
               COALESCE(a.article_text_clean,''),COALESCE(a.word_count,0)
        FROM sec_filing_queue q
        JOIN sec_documents d ON d.accession=q.accession
        JOIN articles a ON a.id=d.article_id
        WHERE d.status='ok' AND a.quality_status='usable'
          AND a.article_text_clean IS NOT NULL
          AND COALESCE(a.canonical_article_id,a.id)=a.id
        ORDER BY q.accepted_at,q.accession,d.document_id
        """
    )
    started = time.monotonic()
    for processed, (
        accession,
        cik,
        filing_date,
        accepted_at,
        form,
        reason,
        description,
        text,
        stored_word_count,
    ) in enumerate(rows, 1):
        event = events.setdefault(
            accession,
            EventAccumulator(accession, int(cik), filing_date, accepted_at, form),
        )
        event.document_count += 1
        event.exhibit99_count += int(reason == "exhibit_99")
        event.primary_count += int(reason == "primary_document")
        event.press_release_count += int(
            reason == "press_release_description"
            or "press release" in description.lower()
        )
        event.char_count += len(text)
        event.word_count += int(stored_word_count) or len(TOKEN_RE.findall(text))
        event.number_count += len(NUMBER_RE.findall(text))
        event.currency_count += len(CURRENCY_RE.findall(text))
        event.percent_count += len(PERCENT_RE.findall(text))
        counts = lexical_counts(text)
        for name, value in counts.items():
            setattr(event, f"{name}_count", getattr(event, f"{name}_count") + value)
        event.item_codes.update(ITEM_RE.findall(text))
        if processed % 5000 == 0:
            elapsed = max(time.monotonic() - started, 0.001)
            print(
                f"Scanned {processed:,} SEC documents across {len(events):,} "
                f"events | {processed / elapsed:.1f} documents/s",
                flush=True,
            )
    return events


def build_metadata(archive_path: Path, output_path: Path) -> dict[str, int]:
    archive_uri = f"file:{archive_path.resolve()}?mode=ro"
    with sqlite3.connect(archive_uri, uri=True) as archive:
        events = load_event_accumulators(archive)
        entity_rows = archive.execute(
            """
            SELECT q.accession,t.ticker,t.company_name,t.sector,q.accepted_at
            FROM sec_filing_queue q
            JOIN sec_filing_tickers t ON t.accession=q.accession
            WHERE q.accession IN (
              SELECT DISTINCT d.accession
              FROM sec_documents d JOIN articles a ON a.id=d.article_id
              WHERE d.status='ok' AND a.quality_status='usable'
                AND COALESCE(a.canonical_article_id,a.id)=a.id
            )
            ORDER BY t.ticker,q.accepted_at,q.accession
            """
        ).fetchall()

    with sqlite3.connect(output_path) as output:
        initialize(output)
        config = {
            "version": "sec-deterministic-v1",
            "archive": archive_path.name,
            "selection_uses_future_returns": False,
        }
        config_json = json.dumps(config, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(config_json.encode()).hexdigest()
        output.execute(
            "INSERT OR IGNORE INTO feature_configs(config_hash,config_json) VALUES(?,?)",
            (digest, config_json),
        )
        for event in events.values():
            after_close, hour = timing_features(event.accepted_at)
            output.execute(
                """
                INSERT INTO sec_event_features(
                    event_id,accession,cik,filing_date,accepted_at,form,
                    item_codes_json,importance_score,after_market_close,
                    accepted_hour_et,document_count,exhibit99_count,primary_count,
                    press_release_count,char_count,word_count,number_count,
                    currency_count,percent_count,positive_count,negative_count,
                    uncertainty_count,litigation_count,constraining_count
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(event_id) DO UPDATE SET
                    item_codes_json=excluded.item_codes_json,
                    importance_score=excluded.importance_score,
                    after_market_close=excluded.after_market_close,
                    accepted_hour_et=excluded.accepted_hour_et,
                    document_count=excluded.document_count,
                    exhibit99_count=excluded.exhibit99_count,
                    primary_count=excluded.primary_count,
                    press_release_count=excluded.press_release_count,
                    char_count=excluded.char_count,word_count=excluded.word_count,
                    number_count=excluded.number_count,
                    currency_count=excluded.currency_count,
                    percent_count=excluded.percent_count,
                    positive_count=excluded.positive_count,
                    negative_count=excluded.negative_count,
                    uncertainty_count=excluded.uncertainty_count,
                    litigation_count=excluded.litigation_count,
                    constraining_count=excluded.constraining_count,
                    updated_at=CURRENT_TIMESTAMP
                """,
                (
                    f"sec:{event.accession}",
                    event.accession,
                    event.cik,
                    event.filing_date,
                    event.accepted_at,
                    event.form,
                    json.dumps(sorted(event.item_codes or ())),
                    importance_score(event),
                    after_close,
                    hour,
                    event.document_count,
                    event.exhibit99_count,
                    event.primary_count,
                    event.press_release_count,
                    event.char_count,
                    event.word_count,
                    event.number_count,
                    event.currency_count,
                    event.percent_count,
                    event.positive_count,
                    event.negative_count,
                    event.uncertainty_count,
                    event.litigation_count,
                    event.constraining_count,
                ),
            )
        previous: dict[str, datetime] = {}
        for accession, ticker, company, sector, accepted_at in entity_rows:
            event_id = f"sec:{accession}"
            if accession not in events:
                continue
            current = parse_accepted_at(accepted_at)
            prior = previous.get(ticker)
            days = (current - prior).total_seconds() / 86400 if prior else None
            output.execute(
                """
                INSERT INTO sec_event_entities(
                    event_id,ticker,company_name,sector,days_since_previous
                ) VALUES(?,?,?,?,?)
                ON CONFLICT(event_id,ticker) DO UPDATE SET
                    company_name=excluded.company_name,sector=excluded.sector,
                    days_since_previous=excluded.days_since_previous
                """,
                (event_id, ticker, company, sector, days),
            )
            previous[ticker] = current
        output.commit()
    return {"events": len(events), "entity_events": len(entity_rows)}


def attach_vectors(
    archive_path: Path,
    embeddings_path: Path,
    output_path: Path,
    progress_every: int = 5000,
) -> dict[str, int]:
    archive_uri = f"file:{archive_path.resolve()}?mode=ro"
    embeddings_uri = f"file:{embeddings_path.resolve()}?mode=ro"
    with sqlite3.connect(embeddings_uri, uri=True) as embeddings:
        config = embeddings.execute(
            """
            SELECT id,dimension,chunking_version FROM embedding_configs
            ORDER BY id DESC LIMIT 1
            """
        ).fetchone()
    if config is None:
        raise RuntimeError("embedding database has no configuration")
    config_id = int(config[0])
    dimension = int(config[1])
    chunking_version = str(config[2])

    with sqlite3.connect(output_path) as output, sqlite3.connect(
        archive_uri, uri=True
    ) as archive:
        initialize(output)
        archive.execute("ATTACH DATABASE ? AS emb", (str(embeddings_path.resolve()),))
        modeled_accessions = {
            row[0] for row in output.execute("SELECT accession FROM sec_event_features")
        }
        article_accessions: dict[int, list[str]] = defaultdict(list)
        for accession, article_id in archive.execute(
            """
            SELECT DISTINCT d.accession,
                   COALESCE(a.canonical_article_id,a.id) AS canonical_article_id
            FROM sec_documents d
            JOIN articles a ON a.id=d.article_id
            WHERE d.status='ok' AND a.quality_status='usable'
            """
        ):
            if accession in modeled_accessions:
                article_accessions[int(article_id)].append(str(accession))

        # Stream vectors in article order. Sorting the original accession join
        # caused SQLite to materialize and sort roughly 1.4 GiB of BLOB payloads
        # before yielding its first row.
        rows = archive.execute(
            """
            SELECT e.article_id,c.token_count,e.vector
            FROM emb.chunk_embeddings e
            JOIN article_chunks c ON c.id=e.chunk_id
            WHERE e.config_id=? AND c.chunking_version=?
            ORDER BY e.article_id,e.chunk_id
            """,
            (config_id, chunking_version),
        )
        totals: dict[str, np.ndarray] = {}
        total_weights: dict[str, int] = defaultdict(int)
        seen = 0
        for article_id, token_count, blob in rows:
            accessions = article_accessions.get(int(article_id))
            if not accessions:
                continue
            vector = np.frombuffer(blob, dtype="<f2").astype(np.float32)
            if vector.size != dimension:
                raise RuntimeError(
                    f"chunk vector dimension {vector.size} differs from {dimension}"
                )
            weight = max(int(token_count), 1)
            weighted = vector * weight
            for accession in accessions:
                if accession in totals:
                    totals[accession] += weighted
                else:
                    totals[accession] = weighted.copy()
                total_weights[accession] += weight
                seen += 1
                if seen % progress_every == 0:
                    print(
                        f"Aggregated {seen:,} event-chunks across "
                        f"{len(totals):,} SEC events",
                        flush=True,
                    )

        stored = 0
        for accession, total in totals.items():
            vector = total / max(total_weights[accession], 1)
            norm = float(np.linalg.norm(vector))
            if norm:
                vector /= norm
            cursor = output.execute(
                """
                UPDATE sec_event_features
                SET embedding_config_id=?,dimension=?,vector=?,
                    updated_at=CURRENT_TIMESTAMP
                WHERE accession=?
                """,
                (
                    config_id,
                    dimension,
                    vector.astype("<f2").tobytes(),
                    accession,
                ),
            )
            stored += max(cursor.rowcount, 0)
        output.commit()

        histories: dict[str, deque[np.ndarray]] = {}
        rows = output.execute(
            """
            SELECT e.event_id,x.ticker,e.vector,e.dimension
            FROM sec_event_features e
            JOIN sec_event_entities x ON x.event_id=e.event_id
            WHERE e.vector IS NOT NULL
            ORDER BY x.ticker,e.accepted_at,e.event_id
            """
        ).fetchall()
        for event_id, ticker, blob, stored_dimension in rows:
            vector = np.frombuffer(blob, dtype="<f2").astype(np.float32)
            if vector.size != stored_dimension:
                raise RuntimeError("stored SEC event vector has invalid dimension")
            prior = histories.get(ticker)
            novelty = None
            if prior:
                maximum_similarity = max(float(np.dot(vector, item)) for item in prior)
                novelty = float(np.clip(1.0 - maximum_similarity, 0.0, 1.0))
            output.execute(
                """
                UPDATE sec_event_entities SET embedding_novelty=?
                WHERE event_id=? AND ticker=?
                """,
                (novelty, event_id, ticker),
            )
            histories.setdefault(ticker, deque(maxlen=20)).append(vector)
        output.commit()
    return {"chunks": seen, "events_with_vectors": stored}


def chronological_split(accepted_at: str) -> str:
    event_date = accepted_at[:10]
    if event_date <= "2022-12-31":
        return "train"
    if event_date <= "2024-12-31":
        return "validation"
    return "test"


def decision_trade_date(
    accepted_at: str, sessions: list[date]
) -> date | None:
    """Return the open after the first market close that can use the filing."""
    local = parse_accepted_at(accepted_at).astimezone(MARKET_TZ)
    event_day = local.date()
    position = bisect.bisect_left(sessions, event_day)
    if position == len(sessions):
        return None
    is_session = sessions[position] == event_day
    before_close = (local.hour, local.minute, local.second) < (16, 0, 0)
    if is_session and before_close:
        decision_position = position
    elif is_session:
        decision_position = position + 1
    else:
        decision_position = position
    trade_position = decision_position + 1
    return sessions[trade_position] if trade_position < len(sessions) else None


def load_calendar(path: Path, date_column: str) -> list[date]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("trading calendar must have a CSV header")
        names = {name.lower(): name for name in reader.fieldnames}
        actual = names.get(date_column.lower())
        if actual is None:
            raise ValueError(
                f"calendar lacks {date_column!r}; found {reader.fieldnames}"
            )
        sessions = sorted(
            {
                date.fromisoformat(row[actual].strip()[:10])
                for row in reader
                if row.get(actual, "").strip()
            }
        )
    if len(sessions) < 2:
        raise ValueError("trading calendar must contain at least two sessions")
    return sessions


def export_daily(
    features_path: Path,
    calendar_path: Path,
    output_path: Path,
    date_column: str = "date",
) -> tuple[int, int]:
    sessions = load_calendar(calendar_path, date_column)
    uri = f"file:{features_path.resolve()}?mode=ro"
    grouped: dict[tuple[date, str], list[sqlite3.Row]] = defaultdict(list)
    deferred = 0
    with sqlite3.connect(uri, uri=True) as db:
        db.row_factory = sqlite3.Row
        for row in db.execute(
            """
            SELECT e.*,x.ticker,x.days_since_previous,x.embedding_novelty
            FROM sec_event_features e
            JOIN sec_event_entities x ON x.event_id=e.event_id
            ORDER BY e.accepted_at,e.event_id,x.ticker
            """
        ):
            trade_day = decision_trade_date(row["accepted_at"], sessions)
            if trade_day is None:
                deferred += 1
                continue
            grouped[(trade_day, row["ticker"])].append(row)

    item_fields = tuple(
        f"sec_item_{item.replace('.', '_')}_count" for item in sorted(ITEM_WEIGHTS)
    )
    feature_fields = (
        "sec_filing_count",
        "sec_8k_count",
        "sec_6k_count",
        "sec_exhibit99_count",
        "sec_document_count",
        "sec_after_close_count",
        "sec_importance_mean",
        "sec_importance_max",
        "sec_embedding_novelty_mean",
        "sec_embedding_novelty_max",
        "sec_days_since_previous",
        "sec_word_count",
        "sec_number_count",
        "sec_currency_count",
        "sec_percent_count",
        "sec_positive_per_1k_words",
        "sec_negative_per_1k_words",
        "sec_uncertainty_per_1k_words",
        "sec_litigation_per_1k_words",
        "sec_constraining_per_1k_words",
        *item_fields,
    )
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("trade_date", "ticker", *feature_fields)
        )
        writer.writeheader()
        for (trade_day, ticker), rows in sorted(grouped.items()):
            total_words = sum(int(row["word_count"]) for row in rows)
            importance = [float(row["importance_score"]) for row in rows]
            novelty = [
                float(row["embedding_novelty"])
                for row in rows
                if row["embedding_novelty"] is not None
            ]
            item_counts = Counter(
                item
                for row in rows
                for item in json.loads(row["item_codes_json"])
            )
            values = {
                "sec_filing_count": len(rows),
                "sec_8k_count": sum(
                    str(row["form"]).upper().startswith("8-K") for row in rows
                ),
                "sec_6k_count": sum(
                    str(row["form"]).upper().startswith("6-K") for row in rows
                ),
                "sec_exhibit99_count": sum(
                    int(row["exhibit99_count"]) for row in rows
                ),
                "sec_document_count": sum(
                    int(row["document_count"]) for row in rows
                ),
                "sec_after_close_count": sum(
                    int(row["after_market_close"]) for row in rows
                ),
                "sec_importance_mean": sum(importance) / len(importance),
                "sec_importance_max": max(importance),
                "sec_embedding_novelty_mean": (
                    sum(novelty) / len(novelty) if novelty else 0.0
                ),
                "sec_embedding_novelty_max": max(novelty, default=0.0),
                "sec_days_since_previous": float(
                    rows[-1]["days_since_previous"] or 0.0
                ),
                "sec_word_count": total_words,
                "sec_number_count": sum(
                    int(row["number_count"]) for row in rows
                ),
                "sec_currency_count": sum(
                    int(row["currency_count"]) for row in rows
                ),
                "sec_percent_count": sum(
                    int(row["percent_count"]) for row in rows
                ),
            }
            for name in LEXICONS:
                values[f"sec_{name}_per_1k_words"] = (
                    1000.0
                    * sum(int(row[f"{name}_count"]) for row in rows)
                    / max(total_words, 1)
                )
            for item, field in zip(sorted(ITEM_WEIGHTS), item_fields, strict=True):
                values[field] = item_counts[item]
            writer.writerow(
                {"trade_date": trade_day.isoformat(), "ticker": ticker, **values}
            )
    return len(grouped), deferred


def select_split(rows: list[dict], quota: int) -> list[dict]:
    if quota <= 0 or not rows:
        return []
    quota = min(quota, len(rows))
    selected: list[dict] = []
    selected_ids: set[str] = set()
    ticker_counts: Counter[str] = Counter()
    years = {row["accepted_at"][:4] for row in rows}
    tickers = {row["ticker"] for row in rows}
    high_quota = min(quota, math.ceil(quota * 0.70))
    ticker_cap = max(2, math.ceil(high_quota / max(len(tickers), 1) * 3))
    year_cap = max(2, math.ceil(high_quota / max(len(years), 1) * 2))
    year_counts: Counter[str] = Counter()

    def add(row: dict, reason: str) -> bool:
        key = f"{row['event_id']}|{row['ticker']}"
        if key in selected_ids:
            return False
        selected_ids.add(key)
        ticker_counts[row["ticker"]] += 1
        year_counts[row["accepted_at"][:4]] += 1
        result = dict(row)
        result["selection_reason"] = reason
        selected.append(result)
        return True

    ranked = sorted(
        rows,
        key=lambda row: (
            -row["priority_score"],
            row["accepted_at"],
            row["event_id"],
            row["ticker"],
        ),
    )
    for row in ranked:
        year = row["accepted_at"][:4]
        if ticker_counts[row["ticker"]] >= ticker_cap or year_counts[year] >= year_cap:
            continue
        add(row, "high_importance")
        if len(selected) >= high_quota:
            break

    groups: dict[tuple[str, str], deque[dict]] = defaultdict(deque)
    for row in ranked:
        key = f"{row['event_id']}|{row['ticker']}"
        if key not in selected_ids:
            groups[(row["accepted_at"][:4], row["sector"] or "Unknown")].append(row)
    group_keys = sorted(groups)
    while len(selected) < quota and group_keys:
        progressed = False
        for group in list(group_keys):
            queue = groups[group]
            while queue:
                row = queue.popleft()
                if add(row, "balanced_coverage"):
                    progressed = True
                    break
            if not queue:
                group_keys.remove(group)
            if len(selected) >= quota:
                break
        if not progressed:
            break
    return selected


def allocate_quotas(limit: int, shares: dict[str, float]) -> dict[str, int]:
    raw = {name: limit * share for name, share in shares.items()}
    quotas = {name: int(math.floor(value)) for name, value in raw.items()}
    remainder = limit - sum(quotas.values())
    order = sorted(
        shares,
        key=lambda name: (-(raw[name] - quotas[name]), name),
    )
    for name in order[:remainder]:
        quotas[name] += 1
    return quotas


def select_candidates(
    features_path: Path,
    output_path: Path,
    limit: int,
    train_share: float,
    validation_share: float,
    pilot_output: Path | None = None,
    pilot_limit: int = 100,
) -> dict[str, int]:
    uri = f"file:{features_path.resolve()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as db:
        db.row_factory = sqlite3.Row
        rows = [
            dict(row)
            for row in db.execute(
                """
                SELECT e.event_id,e.accession,e.accepted_at,e.form,
                       e.item_codes_json,e.importance_score,e.exhibit99_count,
                       e.document_count,e.word_count,x.ticker,x.sector,
                       COALESCE(x.embedding_novelty,0.0) AS embedding_novelty
                FROM sec_event_features e
                JOIN sec_event_entities x ON x.event_id=e.event_id
                ORDER BY e.accepted_at,e.event_id,x.ticker
                """
            )
        ]
    if not rows:
        raise RuntimeError("SEC feature store contains no entity-events")
    for row in rows:
        row["split"] = chronological_split(row["accepted_at"])
        row["priority_score"] = round(
            float(row["importance_score"])
            + 2.0 * float(row["embedding_novelty"])
            + min(math.log1p(int(row["word_count"])) / 20, 0.5),
            6,
        )

    shares = {
        "train": train_share,
        "validation": validation_share,
        "test": 1.0 - train_share - validation_share,
    }
    quotas = allocate_quotas(limit, shares)
    selected = []
    for split in ("train", "validation", "test"):
        selected.extend(
            select_split(
                [row for row in rows if row["split"] == split],
                quotas[split],
            )
        )
    selected.sort(
        key=lambda row: (
            {"train": 0, "validation": 1, "test": 2}[row["split"]],
            -row["priority_score"],
            row["accepted_at"],
            row["event_id"],
            row["ticker"],
        )
    )
    fieldnames = [
        "rank", "event_id", "ticker", "sector", "accepted_at", "split", "form",
        "item_codes_json", "priority_score", "importance_score",
        "embedding_novelty", "exhibit99_count", "document_count", "word_count",
        "selection_reason",
    ]
    def write_rows(path: Path, values: list[dict]) -> None:
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for rank, row in enumerate(values, 1):
                writer.writerow(
                    {
                        "rank": rank,
                        **{
                            name: row.get(name)
                            for name in fieldnames
                            if name != "rank"
                        },
                    }
                )

    write_rows(output_path, selected)
    counts = Counter(row["split"] for row in selected)
    result = {"selected": len(selected), **dict(counts)}
    if pilot_output is not None:
        pilot_quotas = allocate_quotas(pilot_limit, shares)
        pilot = []
        for split in ("train", "validation", "test"):
            pilot.extend(
                select_split(
                    [row for row in rows if row["split"] == split],
                    pilot_quotas[split],
                )
            )
        write_rows(pilot_output, pilot)
        result["pilot"] = len(pilot)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    metadata = subparsers.add_parser("build-metadata")
    metadata.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    metadata.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)

    vectors = subparsers.add_parser("attach-vectors")
    vectors.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    vectors.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS)
    vectors.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    vectors.add_argument("--progress-every", type=int, default=5000)

    select = subparsers.add_parser("select")
    select.add_argument("--features", type=Path, default=DEFAULT_OUTPUT)
    select.add_argument("--output", type=Path, default=DEFAULT_SELECTION)
    select.add_argument("--limit", type=int, default=8000)
    select.add_argument("--train-share", type=float, default=0.65)
    select.add_argument("--validation-share", type=float, default=0.20)
    select.add_argument(
        "--pilot-output",
        type=Path,
        default=Path("sec_deepseek_pilot.csv"),
    )
    select.add_argument("--pilot-limit", type=int, default=100)

    export = subparsers.add_parser("export-daily")
    export.add_argument("--features", type=Path, default=DEFAULT_OUTPUT)
    export.add_argument("--calendar", type=Path, required=True)
    export.add_argument(
        "--output", type=Path, default=Path("sec_deterministic_trading_features.csv")
    )
    export.add_argument("--date-column", default="date")

    args = parser.parse_args()
    if args.command == "build-metadata":
        result = build_metadata(args.archive, args.output)
    elif args.command == "attach-vectors":
        if args.progress_every <= 0:
            parser.error("--progress-every must be positive")
        result = attach_vectors(
            args.archive, args.embeddings, args.output, args.progress_every
        )
    elif args.command == "select":
        test_share = 1.0 - args.train_share - args.validation_share
        if (
            args.limit <= 0
            or args.pilot_limit <= 0
            or args.train_share <= 0
            or args.validation_share <= 0
            or test_share <= 0
        ):
            parser.error("limit and all chronological split shares must be positive")
        result = select_candidates(
            args.features,
            args.output,
            args.limit,
            args.train_share,
            args.validation_share,
            args.pilot_output,
            args.pilot_limit,
        )
    else:
        written, deferred = export_daily(
            args.features, args.calendar, args.output, args.date_column
        )
        result = {"rows": written, "deferred": deferred}
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
