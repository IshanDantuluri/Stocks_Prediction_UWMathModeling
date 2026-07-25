#!/usr/bin/env python3
"""Build same-day news events and high-recall ticker/sector candidates."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sqlite3
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

DEFAULT_ARCHIVE = Path("historical_news.sqlite3")
DEFAULT_EMBEDDINGS = Path("news_embeddings.sqlite3")
DEFAULT_OUTPUT = Path("news_events.sqlite3")
COMPANY_SUFFIX = re.compile(
    r"\s+(?:incorporated|inc|corporation|corp|company|co|limited|ltd|plc|holdings?)\.?$",
    re.IGNORECASE,
)
SHARE_CLASS_SUFFIX = re.compile(
    r"\s*\((?:class|series)\s+[^)]+\)\s*$", re.IGNORECASE
)
AMBIGUOUS_GENERATED_ALIASES = {
    "american",
    "capital",
    "first",
    "general",
    "global",
    "group",
    "international",
    "national",
    "news",
    "united",
}
TOKEN = re.compile(r"[^\W_]+", re.UNICODE)


@dataclass(frozen=True)
class ArticleRecord:
    article_id: int
    event_date: date
    title: str
    domain: str
    vector: np.ndarray


@dataclass(frozen=True)
class EventCluster:
    cluster_id: str
    event_date: date
    article_ids: tuple[int, ...]
    representative_title: str
    centroid: np.ndarray


@dataclass(frozen=True)
class Candidate:
    scope: str
    entity_id: str
    score: float
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class VerifiedLink:
    scope: str
    entity_id: str
    relationship: str
    accepted: bool
    reason: str

    def __post_init__(self) -> None:
        allowed = {
            "direct",
            "probable_indirect",
            "sector",
            "market",
            "contextual",
            "reject",
        }
        if self.relationship not in allowed:
            raise ValueError(f"relationship must be one of {sorted(allowed)}")
        if self.relationship == "reject" and self.accepted:
            raise ValueError("a rejected relationship cannot be accepted")


def normalize_alias(value: str) -> str:
    return " ".join(TOKEN.findall(value.casefold()))


def initialize(db: sqlite3.Connection) -> None:
    db.executescript(
        """
        PRAGMA foreign_keys=ON;
        CREATE TABLE IF NOT EXISTS cluster_configs (
            id INTEGER PRIMARY KEY,
            config_hash TEXT NOT NULL UNIQUE,
            embedding_config_id INTEGER NOT NULL,
            cosine_threshold REAL NOT NULL,
            hard_cosine_threshold REAL NOT NULL,
            title_jaccard_threshold REAL NOT NULL,
            config_json TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS event_clusters (
            cluster_id TEXT PRIMARY KEY,
            config_id INTEGER NOT NULL REFERENCES cluster_configs(id),
            event_date TEXT NOT NULL,
            representative_title TEXT NOT NULL,
            centroid BLOB NOT NULL,
            dimension INTEGER NOT NULL,
            article_count INTEGER NOT NULL,
            source_count INTEGER NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS event_cluster_articles (
            cluster_id TEXT NOT NULL REFERENCES event_clusters(cluster_id)
                ON DELETE CASCADE,
            article_id INTEGER NOT NULL,
            similarity_to_centroid REAL NOT NULL,
            PRIMARY KEY(cluster_id,article_id)
        );
        CREATE TABLE IF NOT EXISTS clustered_days (
            config_id INTEGER NOT NULL REFERENCES cluster_configs(id),
            event_date TEXT NOT NULL,
            article_digest TEXT NOT NULL,
            cluster_count INTEGER NOT NULL,
            PRIMARY KEY(config_id,event_date)
        );
        CREATE TABLE IF NOT EXISTS entities (
            entity_id TEXT PRIMARY KEY,
            company_name TEXT NOT NULL,
            sector TEXT,
            industry TEXT,
            valid_from TEXT,
            valid_to TEXT,
            metadata_json TEXT NOT NULL DEFAULT '{}'
        );
        CREATE TABLE IF NOT EXISTS entity_aliases (
            entity_id TEXT NOT NULL REFERENCES entities(entity_id) ON DELETE CASCADE,
            alias TEXT NOT NULL,
            normalized_alias TEXT NOT NULL,
            alias_kind TEXT NOT NULL,
            valid_from TEXT,
            valid_to TEXT,
            PRIMARY KEY(entity_id,normalized_alias,alias_kind)
        );
        CREATE INDEX IF NOT EXISTS aliases_lookup_idx
            ON entity_aliases(normalized_alias);
        CREATE TABLE IF NOT EXISTS candidate_links (
            cluster_id TEXT NOT NULL REFERENCES event_clusters(cluster_id)
                ON DELETE CASCADE,
            scope TEXT NOT NULL CHECK(scope IN ('ticker','sector','market')),
            entity_id TEXT NOT NULL,
            score REAL NOT NULL,
            reasons_json TEXT NOT NULL,
            generator_version TEXT NOT NULL,
            PRIMARY KEY(cluster_id,scope,entity_id,generator_version)
        );
        CREATE TABLE IF NOT EXISTS verified_links (
            cluster_id TEXT NOT NULL REFERENCES event_clusters(cluster_id)
                ON DELETE CASCADE,
            scope TEXT NOT NULL,
            entity_id TEXT NOT NULL,
            model_id TEXT NOT NULL,
            prompt_version TEXT NOT NULL,
            relationship TEXT NOT NULL,
            accepted INTEGER NOT NULL,
            reason TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY(cluster_id,scope,entity_id,model_id,prompt_version)
        );
        CREATE INDEX IF NOT EXISTS clusters_date_idx
            ON event_clusters(config_id,event_date);
        """
    )


def _active(valid_from: str | None, valid_to: str | None, on_date: date) -> bool:
    value = on_date.isoformat()
    return (valid_from is None or valid_from <= value) and (
        valid_to is None or value <= valid_to
    )


def _pick_column(fieldnames: Sequence[str], *names: str) -> str | None:
    def key(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", value.casefold())

    lookup = {key(name): name for name in fieldnames}
    return next((lookup[key(name)] for name in names if key(name) in lookup), None)


def import_entities(db: sqlite3.Connection, source: Path) -> tuple[int, int]:
    """Import ticker/company/sector/industry/aliases CSV columns."""
    initialize(db)
    with source.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("entity file must be a CSV with a header")
        ticker_col = _pick_column(reader.fieldnames, "ticker", "symbol")
        company_col = _pick_column(
            reader.fieldnames, "company_name", "company", "name", "security"
        )
        if not ticker_col or not company_col:
            raise ValueError("entity CSV requires ticker and company-name columns")
        sector_col = _pick_column(reader.fieldnames, "sector", "gics_sector")
        industry_col = _pick_column(reader.fieldnames, "industry", "gics_industry")
        aliases_col = _pick_column(reader.fieldnames, "aliases", "alias")
        from_col = _pick_column(reader.fieldnames, "valid_from", "start_date")
        to_col = _pick_column(reader.fieldnames, "valid_to", "end_date")
        entity_count = alias_count = 0
        for row in reader:
            ticker = row[ticker_col].strip().upper()
            company = row[company_col].strip()
            if not ticker or not company:
                continue
            valid_from = row.get(from_col, "").strip() or None if from_col else None
            valid_to = row.get(to_col, "").strip() or None if to_col else None
            db.execute(
                """INSERT INTO entities(
                     entity_id,company_name,sector,industry,valid_from,valid_to
                   ) VALUES (?,?,?,?,?,?)
                   ON CONFLICT(entity_id) DO UPDATE SET
                     company_name=excluded.company_name,sector=excluded.sector,
                     industry=excluded.industry,valid_from=excluded.valid_from,
                     valid_to=excluded.valid_to""",
                (
                    ticker,
                    company,
                    row.get(sector_col, "").strip() or None if sector_col else None,
                    row.get(industry_col, "").strip() or None if industry_col else None,
                    valid_from,
                    valid_to,
                ),
            )
            db.execute(
                """DELETE FROM entity_aliases
                   WHERE entity_id=? AND alias_kind IN
                     ('ticker','company','generated_short_name','provided')""",
                (ticker,),
            )
            aliases: set[tuple[str, str]] = {(ticker, "ticker"), (company, "company")}
            without_class = SHARE_CLASS_SUFFIX.sub("", company).strip()
            short = COMPANY_SUFFIX.sub("", without_class).strip()
            normalized_short = normalize_alias(short)
            if (
                len(normalized_short) >= 4
                and normalized_short not in AMBIGUOUS_GENERATED_ALIASES
            ):
                aliases.add((short, "generated_short_name"))
            if aliases_col:
                aliases.update(
                    (value.strip(), "provided")
                    for value in re.split(r"[;|]", row[aliases_col])
                    if value.strip()
                )
            for alias, kind in aliases:
                normalized = normalize_alias(alias)
                if not normalized:
                    continue
                db.execute(
                    """INSERT INTO entity_aliases(
                         entity_id,alias,normalized_alias,alias_kind,valid_from,valid_to
                       ) VALUES (?,?,?,?,?,?)
                       ON CONFLICT(entity_id,normalized_alias,alias_kind) DO UPDATE SET
                         alias=excluded.alias,valid_from=excluded.valid_from,
                         valid_to=excluded.valid_to""",
                    (ticker, alias, normalized, kind, valid_from, valid_to),
                )
                alias_count += 1
            entity_count += 1
    db.commit()
    return entity_count, alias_count


def import_alias_overrides(db: sqlite3.Connection, source: Path) -> tuple[int, int]:
    """Import curated, date-bounded aliases without replacing generated aliases.

    Optional ``alias_kind`` values are preserved.  Current-only knowledge such
    as a product, brand, subsidiary, or executive must have ``valid_from`` so
    it cannot silently leak backwards into historical candidate generation.
    """
    initialize(db)
    imported = skipped = 0
    with source.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("alias file must be a CSV with a header")
        ticker_col = _pick_column(reader.fieldnames, "ticker", "symbol")
        alias_col = _pick_column(reader.fieldnames, "alias", "name")
        from_col = _pick_column(reader.fieldnames, "valid_from", "start_date")
        to_col = _pick_column(reader.fieldnames, "valid_to", "end_date")
        kind_col = _pick_column(reader.fieldnames, "alias_kind", "kind")
        if not ticker_col or not alias_col:
            raise ValueError("alias CSV requires ticker and alias columns")
        known = {row[0] for row in db.execute("SELECT entity_id FROM entities")}
        for row in reader:
            ticker = row[ticker_col].strip().upper()
            alias = row[alias_col].strip()
            normalized = normalize_alias(alias)
            kind = (
                row.get(kind_col, "").strip().casefold().replace(" ", "_")
                if kind_col
                else "override"
            ) or "override"
            if ticker not in known or not normalized:
                skipped += 1
                continue
            valid_from = row.get(from_col, "").strip() or None if from_col else None
            valid_to = row.get(to_col, "").strip() or None if to_col else None
            if kind in {
                "brand",
                "product",
                "subsidiary",
                "executive",
                "prospective",
            } and not valid_from:
                raise ValueError(
                    f"{kind} alias {alias!r} for {ticker} requires valid_from"
                )
            db.execute(
                """INSERT INTO entity_aliases(
                     entity_id,alias,normalized_alias,alias_kind,valid_from,valid_to
                   ) VALUES (?,?,?,?,?,?)
                   ON CONFLICT(entity_id,normalized_alias,alias_kind) DO UPDATE SET
                     alias=excluded.alias,valid_from=excluded.valid_from,
                     valid_to=excluded.valid_to""",
                (ticker, alias, normalized, kind, valid_from, valid_to),
            )
            imported += 1
    db.commit()
    return imported, skipped


def _title_tokens(title: str) -> set[str]:
    return {
        token
        for token in normalize_alias(title).split()
        if len(token) >= 3
    }


def _jaccard(one: set[str], two: set[str]) -> float:
    union = one | two
    return len(one & two) / len(union) if union else 0.0


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(value))
    if not math_is_finite_positive(norm):
        raise ValueError("article vector must be finite and nonzero")
    return value / norm


def math_is_finite_positive(value: float) -> bool:
    return bool(np.isfinite(value) and value > 0)


def cluster_articles(
    articles: Sequence[ArticleRecord],
    cosine_threshold: float = 0.84,
    hard_cosine_threshold: float = 0.92,
    title_jaccard_threshold: float = 0.18,
    config_hash: str = "in-memory",
) -> list[EventCluster]:
    """Deterministic same-day connected-component clustering."""
    if not articles:
        return []
    days = {article.event_date for article in articles}
    if len(days) != 1:
        raise ValueError("cluster_articles accepts exactly one event date")
    ordered = sorted(articles, key=lambda item: item.article_id)
    vectors = [_normalize_vector(item.vector) for item in ordered]
    titles = [_title_tokens(item.title) for item in ordered]
    parent = list(range(len(ordered)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left in range(len(ordered)):
        for right in range(left + 1, len(ordered)):
            similarity = float(vectors[left] @ vectors[right])
            if similarity >= hard_cosine_threshold or (
                similarity >= cosine_threshold
                and _jaccard(titles[left], titles[right]) >= title_jaccard_threshold
            ):
                union(left, right)
    components: dict[int, list[int]] = {}
    for index in range(len(ordered)):
        components.setdefault(find(index), []).append(index)
    clusters = []
    for indices in components.values():
        members = [ordered[index] for index in indices]
        article_ids = tuple(item.article_id for item in members)
        centroid = _normalize_vector(
            np.mean([vectors[index] for index in indices], axis=0)
        )
        identity = (
            f"{config_hash}|{members[0].event_date.isoformat()}|"
            + ",".join(map(str, article_ids))
        )
        cluster_id = hashlib.sha256(identity.encode()).hexdigest()[:24]
        representative = min(
            (item.title for item in members if item.title),
            key=lambda value: (len(value), value),
            default="",
        )
        clusters.append(
            EventCluster(
                cluster_id,
                members[0].event_date,
                article_ids,
                representative,
                centroid,
            )
        )
    return sorted(clusters, key=lambda item: item.article_ids)


def get_or_create_cluster_config(
    db: sqlite3.Connection,
    embedding_config_id: int,
    cosine_threshold: float,
    hard_cosine_threshold: float,
    title_jaccard_threshold: float,
) -> tuple[int, str]:
    config = {
        "embedding_config_id": embedding_config_id,
        "cosine_threshold": cosine_threshold,
        "hard_cosine_threshold": hard_cosine_threshold,
        "title_jaccard_threshold": title_jaccard_threshold,
        "algorithm": "same-day-connected-components-v1",
        "date_semantics": "articles.effective_date-conservative-v1",
    }
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode()).hexdigest()
    db.execute(
        """INSERT OR IGNORE INTO cluster_configs(
             config_hash,embedding_config_id,cosine_threshold,
             hard_cosine_threshold,title_jaccard_threshold,config_json
           ) VALUES (?,?,?,?,?,?)""",
        (
            digest,
            embedding_config_id,
            cosine_threshold,
            hard_cosine_threshold,
            title_jaccard_threshold,
            payload,
        ),
    )
    config_id = db.execute(
        "SELECT id FROM cluster_configs WHERE config_hash=?", (digest,)
    ).fetchone()[0]
    return config_id, digest


def article_metadata(archive: sqlite3.Connection) -> dict[int, tuple[date, str, str]]:
    result = {}
    rows = archive.execute(
        """SELECT root.id,COALESCE(root.effective_date,MIN(e.date)),root.title,root.domain
           FROM articles root
           JOIN articles copies
             ON COALESCE(copies.canonical_article_id,copies.id)=root.id
           JOIN events e ON e.article_id=copies.id
           WHERE root.quality_status='usable'
             AND COALESCE(root.canonical_article_id,root.id)=root.id
           GROUP BY root.id"""
    )
    for article_id, event_date, title, domain in rows:
        result[article_id] = (
            date.fromisoformat(event_date[:10]),
            title or "",
            domain or "",
        )
    return result


def load_article_records(
    archive: sqlite3.Connection,
    embeddings: sqlite3.Connection,
    embedding_config_id: int,
) -> list[ArticleRecord]:
    metadata = article_metadata(archive)
    records: list[ArticleRecord] = []
    current_id = None
    vectors: list[np.ndarray] = []

    def append_current() -> None:
        if current_id is None or current_id not in metadata or not vectors:
            return
        event_date, title, domain = metadata[current_id]
        records.append(
            ArticleRecord(
                current_id,
                event_date,
                title,
                domain,
                _normalize_vector(np.mean(vectors, axis=0)),
            )
        )

    dimension = embeddings.execute(
        "SELECT dimension FROM embedding_configs WHERE id=?",
        (embedding_config_id,),
    ).fetchone()
    if dimension is None:
        raise ValueError("unknown embedding configuration")
    for article_id, blob in embeddings.execute(
        """SELECT article_id,vector FROM chunk_embeddings
           WHERE config_id=? ORDER BY article_id,chunk_id""",
        (embedding_config_id,),
    ):
        if current_id is not None and article_id != current_id:
            append_current()
            vectors = []
        current_id = article_id
        vector = np.frombuffer(blob, dtype="<f2").astype(np.float32)
        if len(vector) != dimension[0]:
            raise ValueError(f"article {article_id} has a malformed embedding")
        vectors.append(vector)
    append_current()
    return records


def persist_day(
    db: sqlite3.Connection,
    config_id: int,
    config_hash: str,
    articles: Sequence[ArticleRecord],
    thresholds: tuple[float, float, float],
) -> tuple[int, bool]:
    if not articles:
        return 0, False
    event_date = articles[0].event_date
    digest = hashlib.sha256(
        ",".join(str(item.article_id) for item in sorted(articles, key=lambda x: x.article_id))
        .encode()
    ).hexdigest()
    existing = db.execute(
        """SELECT article_digest,cluster_count FROM clustered_days
           WHERE config_id=? AND event_date=?""",
        (config_id, event_date.isoformat()),
    ).fetchone()
    if existing and existing[0] == digest:
        return existing[1], False
    old_ids = [
        row[0]
        for row in db.execute(
            "SELECT cluster_id FROM event_clusters WHERE config_id=? AND event_date=?",
            (config_id, event_date.isoformat()),
        )
    ]
    db.executemany(
        "DELETE FROM event_clusters WHERE cluster_id=?",
        ((cluster_id,) for cluster_id in old_ids),
    )
    clusters = cluster_articles(articles, *thresholds, config_hash=config_hash)
    domains = {article.article_id: article.domain for article in articles}
    vectors = {article.article_id: _normalize_vector(article.vector) for article in articles}
    for cluster in clusters:
        db.execute(
            """INSERT INTO event_clusters(
                 cluster_id,config_id,event_date,representative_title,centroid,
                 dimension,article_count,source_count
               ) VALUES (?,?,?,?,?,?,?,?)""",
            (
                cluster.cluster_id,
                config_id,
                cluster.event_date.isoformat(),
                cluster.representative_title,
                np.asarray(cluster.centroid, dtype="<f2").tobytes(),
                len(cluster.centroid),
                len(cluster.article_ids),
                len({domains[value] for value in cluster.article_ids if domains[value]}),
            ),
        )
        db.executemany(
            "INSERT INTO event_cluster_articles VALUES (?,?,?)",
            (
                (
                    cluster.cluster_id,
                    article_id,
                    float(vectors[article_id] @ cluster.centroid),
                )
                for article_id in cluster.article_ids
            ),
        )
    db.execute(
        """INSERT INTO clustered_days VALUES (?,?,?,?)
           ON CONFLICT(config_id,event_date) DO UPDATE SET
             article_digest=excluded.article_digest,
             cluster_count=excluded.cluster_count""",
        (config_id, event_date.isoformat(), digest, len(clusters)),
    )
    db.commit()
    return len(clusters), True


def candidates_for_text(
    db: sqlite3.Connection,
    text: str,
    event_date: date,
    generator_version: str = "exact-alias-v1",
) -> list[Candidate]:
    """Return ticker candidates plus sectors inherited from ticker matches."""
    normalized_text = f" {normalize_alias(text)} "
    original = text
    found: dict[tuple[str, str], dict[str, object]] = {}
    rows = db.execute(
        """SELECT a.entity_id,a.alias,a.normalized_alias,a.alias_kind,
                  a.valid_from,a.valid_to,e.sector,e.valid_from,e.valid_to
           FROM entity_aliases a JOIN entities e ON e.entity_id=a.entity_id"""
    )
    for (
        ticker,
        alias,
        normalized,
        kind,
        alias_from,
        alias_to,
        sector,
        entity_from,
        entity_to,
    ) in rows:
        if not _active(alias_from, alias_to, event_date) or not _active(
            entity_from, entity_to, event_date
        ):
            continue
        if kind == "ticker":
            escaped = re.escape(alias)
            alternatives = rf"\${escaped}"
            if len(alias) >= 3:
                alternatives += rf"|{escaped}"
            matched = bool(
                re.search(
                    rf"(?<![A-Za-z0-9])(?:{alternatives})(?![A-Za-z0-9])",
                    original,
                )
            )
            score = 0.92
        else:
            matched = f" {normalized} " in normalized_text
            score = {
                "company": 1.0,
                "provided": 1.0,
                "override": 0.98,
                "sec_legal_name": 0.99,
                "subsidiary": 0.94,
                "brand": 0.90,
                "product": 0.86,
                "executive": 0.82,
            }.get(
                kind,
                0.99 if kind.startswith("sec_former_name") else 0.90,
            )
        if not matched:
            continue
        key = ("ticker", ticker)
        bucket = found.setdefault(key, {"score": 0.0, "reasons": set()})
        bucket["score"] = max(float(bucket["score"]), score)
        bucket["reasons"].add(f"{kind} match: {alias}")
        if sector:
            sector_key = ("sector", sector)
            sector_bucket = found.setdefault(
                sector_key, {"score": 0.65, "reasons": set()}
            )
            sector_bucket["reasons"].add(f"inherited from ticker candidate {ticker}")
    return sorted(
        (
            Candidate(scope, entity_id, float(value["score"]), tuple(sorted(value["reasons"])))
            for (scope, entity_id), value in found.items()
        ),
        key=lambda item: (-item.score, item.scope, item.entity_id),
    )


def cluster_text(
    archive: sqlite3.Connection, article_ids: Sequence[int], max_chars: int = 12000
) -> str:
    if not article_ids:
        return ""
    marks = ",".join("?" for _ in article_ids)
    rows = archive.execute(
        f"""SELECT a.title,a.article_text_clean FROM articles a
            WHERE a.id IN ({marks}) ORDER BY a.id""",
        tuple(article_ids),
    )
    parts = []
    remaining = max_chars
    for title, body in rows:
        value = f"{title or ''}\n{body or ''}".strip()
        if not value:
            continue
        parts.append(value[:remaining])
        remaining -= min(len(value), remaining)
        if remaining <= 0:
            break
    return "\n\n---\n\n".join(parts)


def verification_payload(
    cluster_id: str,
    event_date: date,
    title: str,
    text: str,
    candidates: Sequence[Candidate],
) -> dict:
    return {
        "task": (
            "Identify materially affected entities. Prefer rejecting contextual "
            "mentions; return additional named companies separately for registry resolution."
        ),
        "cluster_id": cluster_id,
        "event_date": event_date.isoformat(),
        "title": title,
        "event_text": text,
        "candidates": [asdict(candidate) for candidate in candidates],
        "allowed_relationships": (
            "direct",
            "probable_indirect",
            "sector",
            "contextual",
            "reject",
        ),
        "output": {
            "links": "VerifiedLink[]",
            "additional_company_names": "string[]",
            "event_summary": "concise facts known on event_date",
        },
    }


def save_candidates(
    db: sqlite3.Connection,
    cluster_id: str,
    candidates: Iterable[Candidate],
    generator_version: str = "exact-alias-v1",
) -> int:
    rows = [
        (
            cluster_id,
            item.scope,
            item.entity_id,
            item.score,
            json.dumps(item.reasons),
            generator_version,
        )
        for item in candidates
    ]
    db.execute(
        "DELETE FROM candidate_links WHERE cluster_id=? AND generator_version=?",
        (cluster_id, generator_version),
    )
    db.executemany(
        """INSERT INTO candidate_links VALUES (?,?,?,?,?,?)
           ON CONFLICT(cluster_id,scope,entity_id,generator_version) DO UPDATE SET
             score=excluded.score,reasons_json=excluded.reasons_json""",
        rows,
    )
    return len(rows)


def load_saved_candidates(
    db: sqlite3.Connection,
    cluster_id: str,
    generator_version: str = "exact-alias-v1",
) -> list[Candidate]:
    """Load a cluster's previously generated candidates without rematching text."""
    rows = db.execute(
        """SELECT scope,entity_id,score,reasons_json
           FROM candidate_links
           WHERE cluster_id=? AND generator_version=?
           ORDER BY score DESC,scope,entity_id""",
        (cluster_id, generator_version),
    )
    return [
        Candidate(scope, entity_id, float(score), tuple(json.loads(reasons)))
        for scope, entity_id, score, reasons in rows
    ]


def run_clustering(args) -> None:
    archive_uri = f"file:{args.archive.resolve()}?mode=ro"
    embedding_uri = f"file:{args.embeddings.resolve()}?mode=ro"
    with sqlite3.connect(archive_uri, uri=True) as archive, sqlite3.connect(
        embedding_uri, uri=True
    ) as embeddings, sqlite3.connect(args.output) as output:
        initialize(output)
        embedding_config_id = args.embedding_config_id
        if embedding_config_id is None:
            row = embeddings.execute(
                "SELECT id FROM embedding_configs ORDER BY id DESC LIMIT 1"
            ).fetchone()
            if row is None:
                raise RuntimeError("embedding database is empty")
            embedding_config_id = row[0]
        config_id, config_hash = get_or_create_cluster_config(
            output,
            embedding_config_id,
            args.cosine_threshold,
            args.hard_cosine_threshold,
            args.title_jaccard_threshold,
        )
        print(
            f"Streaming chunk vectors for embedding config {embedding_config_id}...",
            flush=True,
        )
        records = load_article_records(archive, embeddings, embedding_config_id)
        print(f"Prepared {len(records):,} article vectors.", flush=True)
        by_day: dict[date, list[ArticleRecord]] = {}
        for record in records:
            by_day.setdefault(record.event_date, []).append(record)
        selected_days = sorted(by_day)
        if args.limit_days is not None:
            selected_days = selected_days[: args.limit_days]
        changed = skipped = cluster_count = 0
        thresholds = (
            args.cosine_threshold,
            args.hard_cosine_threshold,
            args.title_jaccard_threshold,
        )
        for index, day in enumerate(selected_days, 1):
            count, was_changed = persist_day(
                output, config_id, config_hash, by_day[day], thresholds
            )
            cluster_count += count
            changed += int(was_changed)
            skipped += int(not was_changed)
            if index % 100 == 0 or index == len(selected_days):
                print(
                    f"Clustered {index:,}/{len(selected_days):,} days | "
                    f"{cluster_count:,} events | changed {changed:,} | skipped {skipped:,}",
                    flush=True,
                )


def run_candidate_generation(args) -> None:
    archive_uri = f"file:{args.archive.resolve()}?mode=ro"
    with sqlite3.connect(archive_uri, uri=True) as archive, sqlite3.connect(
        args.output
    ) as output:
        initialize(output)
        config_id = args.config_id
        if config_id is None:
            row = output.execute(
                "SELECT id FROM cluster_configs ORDER BY id DESC LIMIT 1"
            ).fetchone()
            if row is None:
                raise RuntimeError("no cluster configuration exists; run cluster first")
            config_id = row[0]
        if args.reuse_candidates:
            clusters = output.execute(
                """SELECT ec.cluster_id,ec.event_date,ec.representative_title
                   FROM event_clusters ec
                   WHERE ec.config_id=?
                     AND EXISTS (
                       SELECT 1 FROM candidate_links cl
                       WHERE cl.cluster_id=ec.cluster_id
                         AND cl.generator_version='exact-alias-v1'
                         AND cl.scope='ticker'
                     )
                   ORDER BY ec.event_date,ec.cluster_id""",
                (config_id,),
            ).fetchall()
        else:
            clusters = output.execute(
                """SELECT cluster_id,event_date,representative_title
                   FROM event_clusters WHERE config_id=?
                   ORDER BY event_date,cluster_id""",
                (config_id,),
            ).fetchall()
        if args.sample is not None and args.sample < len(clusters):
            positions = np.linspace(
                0, len(clusters) - 1, num=args.sample, dtype=np.int64
            )
            clusters = [clusters[int(position)] for position in positions]
        if args.limit is not None:
            clusters = clusters[: args.limit]
        jsonl_handle = args.jsonl.open("w") if args.jsonl else None
        total_candidates = 0
        positive_payloads = []
        try:
            for index, (cluster_id, event_date, title) in enumerate(clusters, 1):
                article_ids = [
                    row[0]
                    for row in output.execute(
                        """SELECT article_id FROM event_cluster_articles
                           WHERE cluster_id=? ORDER BY article_id""",
                        (cluster_id,),
                    )
                ]
                text = cluster_text(archive, article_ids, args.max_chars)
                if args.reuse_candidates:
                    candidates = load_saved_candidates(output, cluster_id)
                    total_candidates += len(candidates)
                else:
                    candidates = candidates_for_text(
                        output, f"{title}\n{text}", date.fromisoformat(event_date)
                    )
                    total_candidates += save_candidates(output, cluster_id, candidates)
                payload = verification_payload(
                    cluster_id,
                    date.fromisoformat(event_date),
                    title,
                    text,
                    candidates,
                )
                if args.ticker_positive_sample is not None:
                    if any(item.scope == "ticker" for item in candidates):
                        positive_payloads.append(payload)
                elif jsonl_handle:
                    jsonl_handle.write(json.dumps(payload) + "\n")
                if index % 100 == 0 or index == len(clusters):
                    output.commit()
                    print(
                        f"Generated candidates for {index:,}/{len(clusters):,} "
                        f"events | {total_candidates:,} candidate links",
                        flush=True,
                    )
            if args.ticker_positive_sample is not None:
                count = min(args.ticker_positive_sample, len(positive_payloads))
                if count:
                    positions = np.linspace(
                        0, len(positive_payloads) - 1, num=count, dtype=np.int64
                    )
                    selected = [
                        positive_payloads[int(position)] for position in positions
                    ]
                else:
                    selected = []
                for payload in selected:
                    jsonl_handle.write(json.dumps(payload) + "\n")
                print(
                    f"Wrote {len(selected):,} date-stratified ticker-positive events "
                    f"from {len(positive_payloads):,} matching events to {args.jsonl}",
                    flush=True,
                )
        finally:
            if jsonl_handle:
                jsonl_handle.close()
        output.commit()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init", help="initialize event/entity storage")
    init.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    entities = commands.add_parser("import-entities", help="import a ticker registry CSV")
    entities.add_argument("csv", type=Path)
    entities.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    aliases = commands.add_parser(
        "import-aliases", help="import curated dated ticker aliases"
    )
    aliases.add_argument("csv", type=Path)
    aliases.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    cluster = commands.add_parser("cluster", help="cluster canonical articles by news date")
    cluster.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    cluster.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS)
    cluster.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    cluster.add_argument("--embedding-config-id", type=int)
    cluster.add_argument("--cosine-threshold", type=float, default=0.84)
    cluster.add_argument("--hard-cosine-threshold", type=float, default=0.92)
    cluster.add_argument("--title-jaccard-threshold", type=float, default=0.18)
    cluster.add_argument("--limit-days", type=int)
    candidates = commands.add_parser(
        "generate-candidates",
        help="match entity aliases and optionally write LLM verification JSONL",
    )
    candidates.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    candidates.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    candidates.add_argument("--config-id", type=int)
    candidates.add_argument("--limit", type=int)
    candidates.add_argument(
        "--sample",
        type=int,
        help="evenly sample this many events across the full date range",
    )
    candidates.add_argument(
        "--ticker-positive-sample",
        type=int,
        help=(
            "scan all events and write this many date-stratified events having "
            "at least one ticker candidate"
        ),
    )
    candidates.add_argument("--max-chars", type=int, default=12000)
    candidates.add_argument("--jsonl", type=Path)
    candidates.add_argument(
        "--reuse-candidates",
        action="store_true",
        help=(
            "export ticker-positive payloads from stored candidate links without "
            "rescanning or rematching every event"
        ),
    )
    args = parser.parse_args()
    if args.command == "cluster":
        run_clustering(args)
        return
    if args.command == "generate-candidates":
        if args.reuse_candidates:
            if args.jsonl is None:
                parser.error("--reuse-candidates requires --jsonl")
            if (
                args.sample is not None
                or args.limit is not None
                or args.ticker_positive_sample is not None
            ):
                parser.error(
                    "--reuse-candidates cannot be combined with sampling or limits"
                )
        if args.ticker_positive_sample is not None:
            if args.ticker_positive_sample <= 0 or args.jsonl is None:
                parser.error(
                    "--ticker-positive-sample requires a positive value and --jsonl"
                )
            if args.sample is not None or args.limit is not None:
                parser.error(
                    "--ticker-positive-sample cannot be combined with --sample or --limit"
                )
        run_candidate_generation(args)
        return
    with sqlite3.connect(args.output) as db:
        initialize(db)
        if args.command == "init":
            print(f"Initialized {args.output}")
        elif args.command == "import-entities":
            entities_count, aliases_count = import_entities(db, args.csv)
            print(
                f"Imported {entities_count:,} entities and "
                f"{aliases_count:,} alias records"
            )
        else:
            imported, skipped = import_alias_overrides(db, args.csv)
            print(f"Imported {imported:,} alias overrides; skipped {skipped:,}")


if __name__ == "__main__":
    main()
