#!/usr/bin/env python3
"""Collect prospective news with auditable, leakage-safe timestamps.

The first adapter uses Alpha Vantage's market-wide ``NEWS_SENTIMENT`` feed, so
one request can cover many tickers. Raw responses are retained as compressed
JSON. ``materialize`` adds URLs or provider-supplied bodies to the historical
archive while preserving exact first-retrieval timestamps and ticker tags.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urlparse

import requests

from news_archive import (
    CLEANING_VERSION,
    clean_article,
    content_hash,
    initialize as initialize_archive,
    simhash,
    update_effective_dates,
)

DEFAULT_DATABASE = Path("prospective_news.sqlite3")
DEFAULT_RAW_DIR = Path("prospective_news_raw")
DEFAULT_ARCHIVE = Path("historical_news.sqlite3")
ALPHA_URL = "https://www.alphavantage.co/query"
ALPHA_TIME = re.compile(r"^(\d{8})T(\d{4,6})$")
SECRET_QUERY = re.compile(r"(?i)(apikey|api_key|token)=([^&\s]+)")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key and key not in os.environ:
            os.environ[key] = value.strip().strip("'\"")


def redact_secrets(value: str) -> str:
    return SECRET_QUERY.sub(r"\1=REDACTED", value)


def connect(path: Path) -> sqlite3.Connection:
    db = sqlite3.connect(path, timeout=60)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA foreign_keys=ON")
    db.execute("PRAGMA busy_timeout=60000")
    return db


def initialize(db: sqlite3.Connection) -> None:
    db.executescript(
        """
        CREATE TABLE IF NOT EXISTS provider_runs (
            run_id INTEGER PRIMARY KEY,
            provider TEXT NOT NULL,
            started_at TEXT NOT NULL,
            completed_at TEXT,
            status TEXT NOT NULL,
            request_parameters_json TEXT NOT NULL,
            article_count INTEGER NOT NULL DEFAULT 0,
            error TEXT
        );
        CREATE TABLE IF NOT EXISTS raw_payloads (
            payload_id INTEGER PRIMARY KEY,
            run_id INTEGER NOT NULL REFERENCES provider_runs(run_id),
            provider TEXT NOT NULL,
            object_key TEXT NOT NULL,
            raw_path TEXT NOT NULL,
            retrieved_at TEXT NOT NULL,
            content_sha256 TEXT NOT NULL,
            compressed_bytes INTEGER NOT NULL,
            UNIQUE(provider,object_key)
        );
        CREATE TABLE IF NOT EXISTS articles (
            provider TEXT NOT NULL,
            provider_id TEXT NOT NULL,
            url TEXT NOT NULL,
            title TEXT,
            summary TEXT,
            body TEXT,
            source_name TEXT,
            published_at TEXT,
            updated_at TEXT,
            first_retrieved_at TEXT NOT NULL,
            last_retrieved_at TEXT NOT NULL,
            content_sha256 TEXT,
            raw_payload_id INTEGER REFERENCES raw_payloads(payload_id),
            PRIMARY KEY(provider,provider_id)
        );
        CREATE INDEX IF NOT EXISTS prospective_articles_url_idx ON articles(url);
        CREATE INDEX IF NOT EXISTS prospective_articles_time_idx
            ON articles(first_retrieved_at,published_at);
        CREATE TABLE IF NOT EXISTS article_tickers (
            provider TEXT NOT NULL,
            provider_id TEXT NOT NULL,
            ticker TEXT NOT NULL,
            relevance REAL,
            sentiment REAL,
            sentiment_label TEXT,
            is_primary INTEGER,
            PRIMARY KEY(provider,provider_id,ticker),
            FOREIGN KEY(provider,provider_id)
                REFERENCES articles(provider,provider_id) ON DELETE CASCADE
        );
        """
    )
    db.commit()


def parse_alpha_time(value: str | None) -> str | None:
    if not value:
        return None
    match = ALPHA_TIME.match(value.strip())
    if not match:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc).isoformat(timespec="seconds")
        except ValueError:
            return None
    day, clock = match.groups()
    fmt = "%Y%m%d%H%M%S" if len(clock) == 6 else "%Y%m%d%H%M"
    parsed = datetime.strptime(day + clock, fmt).replace(tzinfo=timezone.utc)
    return parsed.isoformat(timespec="seconds")


def safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def provider_id(item: dict[str, Any]) -> str:
    explicit = str(item.get("id") or item.get("uuid") or "").strip()
    if explicit:
        return explicit
    identity = "\n".join(
        str(item.get(key) or "") for key in ("url", "title", "time_published")
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def validate_alpha_payload(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        raise RuntimeError("Alpha Vantage returned a non-object response")
    detail = (
        payload.get("Error Message")
        or payload.get("Information")
        or payload.get("Note")
    )
    if detail:
        raise RuntimeError(f"Alpha Vantage rejected the request: {str(detail)[:300]}")
    feed = payload.get("feed")
    if not isinstance(feed, list):
        raise RuntimeError("Alpha Vantage response has no news feed")
    return [item for item in feed if isinstance(item, dict)]


def write_raw_payload(
    db: sqlite3.Connection,
    raw_dir: Path,
    run_id: int,
    provider: str,
    payload: Any,
    retrieved_at: str,
) -> int:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    target = raw_dir / provider / retrieved_at[:10] / f"{digest}.json.gz"
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        temporary = target.with_suffix(".tmp")
        with gzip.open(temporary, "wb") as handle:
            handle.write(encoded)
        temporary.replace(target)
    object_key = f"news/{retrieved_at}/{digest[:16]}"
    db.execute(
        """INSERT OR IGNORE INTO raw_payloads(
             run_id,provider,object_key,raw_path,retrieved_at,
             content_sha256,compressed_bytes
           ) VALUES (?,?,?,?,?,?,?)""",
        (
            run_id,
            provider,
            object_key,
            str(target),
            retrieved_at,
            digest,
            target.stat().st_size,
        ),
    )
    return db.execute(
        "SELECT payload_id FROM raw_payloads WHERE provider=? AND object_key=?",
        (provider, object_key),
    ).fetchone()[0]


def upsert_alpha_feed(
    db: sqlite3.Connection,
    feed: list[dict[str, Any]],
    retrieved_at: str,
    raw_payload_id: int,
) -> int:
    written = 0
    for item in feed:
        url = str(item.get("url") or "").strip()
        title = str(item.get("title") or "").strip() or None
        if not url or not title:
            continue
        item_id = provider_id(item)
        body_value = item.get("body") or item.get("content")
        body = str(body_value).strip() if body_value else None
        summary = str(item.get("summary") or "").strip() or None
        published_at = parse_alpha_time(item.get("time_published"))
        updated_at = parse_alpha_time(item.get("time_updated"))
        digest = hashlib.sha256(body.encode("utf-8")).hexdigest() if body else None
        db.execute(
            """INSERT INTO articles(
                 provider,provider_id,url,title,summary,body,source_name,
                 published_at,updated_at,first_retrieved_at,last_retrieved_at,
                 content_sha256,raw_payload_id
               ) VALUES ('alpha_vantage',?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(provider,provider_id) DO UPDATE SET
                 url=excluded.url,title=excluded.title,summary=excluded.summary,
                 body=COALESCE(excluded.body,articles.body),
                 source_name=excluded.source_name,
                 published_at=COALESCE(excluded.published_at,articles.published_at),
                 updated_at=COALESCE(excluded.updated_at,articles.updated_at),
                 last_retrieved_at=excluded.last_retrieved_at,
                 content_sha256=COALESCE(excluded.content_sha256,articles.content_sha256),
                 raw_payload_id=excluded.raw_payload_id""",
            (
                item_id,
                url,
                title,
                summary,
                body,
                str(item.get("source") or "").strip() or None,
                published_at,
                updated_at,
                retrieved_at,
                retrieved_at,
                digest,
                raw_payload_id,
            ),
        )
        db.execute(
            "DELETE FROM article_tickers WHERE provider='alpha_vantage' AND provider_id=?",
            (item_id,),
        )
        for index, tag in enumerate(item.get("ticker_sentiment") or []):
            if not isinstance(tag, dict):
                continue
            ticker = str(tag.get("ticker") or "").strip().upper()
            if not ticker:
                continue
            db.execute(
                """INSERT OR REPLACE INTO article_tickers VALUES
                   ('alpha_vantage',?,?,?,?,?,?)""",
                (
                    item_id,
                    ticker,
                    safe_float(tag.get("relevance_score")),
                    safe_float(tag.get("ticker_sentiment_score")),
                    str(tag.get("ticker_sentiment_label") or "").strip() or None,
                    int(index == 0),
                ),
            )
        written += 1
    return written


def alpha_time_parameter(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y%m%dT%H%M")


def sync_alpha(args: argparse.Namespace) -> None:
    load_env_file(args.env_file)
    api_key = os.environ.get("ALPHA_VANTAGE_API_KEY", "")
    if not api_key or api_key == "replace_me":
        raise RuntimeError("Set ALPHA_VANTAGE_API_KEY in .env")
    time_from = args.time_from or alpha_time_parameter(
        datetime.now(timezone.utc) - timedelta(hours=args.hours_back)
    )
    parameters: dict[str, Any] = {
        "function": "NEWS_SENTIMENT",
        "sort": "LATEST",
        "limit": args.limit,
        "time_from": time_from,
        "apikey": api_key,
    }
    if args.tickers:
        parameters["tickers"] = args.tickers
    recorded = {key: value for key, value in parameters.items() if key != "apikey"}
    with connect(args.database) as db:
        initialize(db)
        cursor = db.execute(
            """INSERT INTO provider_runs(
                 provider,started_at,status,request_parameters_json)
               VALUES ('alpha_vantage',?,'running',?)""",
            (utc_now(), json.dumps(recorded, sort_keys=True)),
        )
        run_id = cursor.lastrowid
        db.commit()
        try:
            try:
                response = requests.get(
                    f"{ALPHA_URL}?{urlencode(parameters)}", timeout=args.timeout
                )
                response.raise_for_status()
            except requests.RequestException as exc:
                raise RuntimeError(
                    f"Alpha Vantage request failed ({type(exc).__name__})"
                ) from None
            payload = response.json()
            feed = validate_alpha_payload(payload)
            retrieved_at = utc_now()
            payload_id = write_raw_payload(
                db, args.raw_dir, run_id, "alpha_vantage", payload, retrieved_at
            )
            written = upsert_alpha_feed(db, feed, retrieved_at, payload_id)
            db.execute(
                """UPDATE provider_runs SET completed_at=?,status='ok',
                   article_count=? WHERE run_id=?""",
                (utc_now(), written, run_id),
            )
            db.commit()
        except Exception as exc:
            db.execute(
                """UPDATE provider_runs SET completed_at=?,status='error',error=?
                   WHERE run_id=?""",
                (
                    utc_now(),
                    redact_secrets(f"{type(exc).__name__}: {str(exc)[:500]}"),
                    run_id,
                ),
            )
            db.commit()
            raise
    print(
        f"Collected {written:,} prospective articles from Alpha Vantage "
        f"(time_from={time_from}; one API request)",
        flush=True,
    )


def initialize_archive_provenance(db: sqlite3.Connection) -> None:
    db.executescript(
        """
        CREATE TABLE IF NOT EXISTS prospective_article_sources (
            provider TEXT NOT NULL,
            provider_id TEXT NOT NULL,
            article_id INTEGER NOT NULL REFERENCES articles(id),
            event_id INTEGER REFERENCES events(id),
            published_at TEXT,
            updated_at TEXT,
            first_retrieved_at TEXT NOT NULL,
            last_retrieved_at TEXT NOT NULL,
            available_at TEXT NOT NULL,
            raw_payload_path TEXT,
            tagged_tickers_json TEXT NOT NULL DEFAULT '[]',
            PRIMARY KEY(provider,provider_id)
        );
        CREATE INDEX IF NOT EXISTS prospective_available_idx
            ON prospective_article_sources(available_at);
        """
    )


def clean_imported_body(
    archive: sqlite3.Connection, article_id: int, raw: str, title: str | None
) -> None:
    item = clean_article(raw, title)
    digest = content_hash(item["text"]) if item["text"] else None
    fingerprint = simhash(item["text"]) if item["text"] else None
    canonical = article_id
    if digest:
        row = archive.execute(
            """SELECT COALESCE(canonical_article_id,id) FROM articles
               WHERE content_hash=? AND id<>? ORDER BY id LIMIT 1""",
            (digest, article_id),
        ).fetchone()
        if row:
            canonical = row[0]
    archive.execute(
        """UPDATE articles SET title=?,article_text_clean=?,cleaning_version=?,
           word_count=?,quality_score=?,quality_status=?,quality_reasons=?,
           content_hash=?,simhash=?,canonical_article_id=? WHERE id=?""",
        (
            item["title"],
            item["text"],
            CLEANING_VERSION,
            item["word_count"],
            item["score"],
            item["status"],
            json.dumps(item["reasons"]),
            digest,
            f"{fingerprint:016x}" if fingerprint is not None else None,
            canonical,
            article_id,
        ),
    )
    if fingerprint is not None:
        for band in range(4):
            bucket = (fingerprint >> (band * 16)) & 0xFFFF
            archive.execute(
                "INSERT OR IGNORE INTO article_fingerprint_bands VALUES (?,?,?)",
                (band, bucket, article_id),
            )


def materialize(source_db: Path, archive_path: Path) -> tuple[int, int]:
    source_uri = f"file:{source_db.resolve()}?mode=ro"
    imported = body_count = 0
    with sqlite3.connect(source_uri, uri=True) as source, connect(archive_path) as archive:
        initialize_archive(archive)
        initialize_archive_provenance(archive)
        rows = source.execute(
            """SELECT a.provider,a.provider_id,a.url,a.title,a.body,a.published_at,
                      a.updated_at,a.first_retrieved_at,a.last_retrieved_at,
                      rp.raw_path
               FROM articles a LEFT JOIN raw_payloads rp
                 ON rp.payload_id=a.raw_payload_id
               ORDER BY a.first_retrieved_at,a.provider,a.provider_id"""
        ).fetchall()
        for (
            provider,
            item_id,
            url,
            title,
            body,
            published_at,
            updated_at,
            first_retrieved_at,
            last_retrieved_at,
            raw_path,
        ) in rows:
            domain = urlparse(url).netloc.lower().removeprefix("www.")
            archive.execute(
                """INSERT INTO articles(
                     source_url,domain,title_raw,published_at,status,
                     article_text_raw,article_text,fetched_at)
                   VALUES (?,?,?,?,?,?,?,?)
                   ON CONFLICT(source_url) DO UPDATE SET
                     title_raw=COALESCE(articles.title_raw,excluded.title_raw),
                     published_at=COALESCE(articles.published_at,excluded.published_at),
                     status=CASE WHEN excluded.article_text_raw IS NOT NULL
                                 THEN 'ok' ELSE articles.status END,
                     article_text_raw=COALESCE(
                         articles.article_text_raw,excluded.article_text_raw),
                     article_text=COALESCE(
                         articles.article_text,excluded.article_text),
                     fetched_at=COALESCE(articles.fetched_at,excluded.fetched_at)""",
                (
                    url,
                    domain,
                    title,
                    published_at,
                    "ok" if body else "pending",
                    body,
                    body,
                    first_retrieved_at if body else None,
                ),
            )
            article_id = archive.execute(
                "SELECT id FROM articles WHERE source_url=?", (url,)
            ).fetchone()[0]
            existing = archive.execute(
                """SELECT event_id FROM prospective_article_sources
                   WHERE provider=? AND provider_id=?""",
                (provider, item_id),
            ).fetchone()
            available_at = first_retrieved_at
            if existing and existing[0]:
                event_id = existing[0]
            else:
                cursor = archive.execute(
                    """INSERT INTO events(
                         date,event_category,primary_actor,article_id)
                       VALUES (?,'prospective_news',?,?)""",
                    (available_at[:10], provider, article_id),
                )
                event_id = cursor.lastrowid
            tickers = [
                row[0]
                for row in source.execute(
                    """SELECT ticker FROM article_tickers
                       WHERE provider=? AND provider_id=? ORDER BY ticker""",
                    (provider, item_id),
                )
            ]
            archive.execute(
                """INSERT INTO prospective_article_sources(
                     provider,provider_id,article_id,event_id,published_at,
                     updated_at,first_retrieved_at,last_retrieved_at,available_at,
                     raw_payload_path,tagged_tickers_json)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(provider,provider_id) DO UPDATE SET
                     article_id=excluded.article_id,event_id=excluded.event_id,
                     published_at=excluded.published_at,updated_at=excluded.updated_at,
                     last_retrieved_at=excluded.last_retrieved_at,
                     raw_payload_path=excluded.raw_payload_path,
                     tagged_tickers_json=excluded.tagged_tickers_json""",
                (
                    provider,
                    item_id,
                    article_id,
                    event_id,
                    published_at,
                    updated_at,
                    first_retrieved_at,
                    last_retrieved_at,
                    available_at,
                    raw_path,
                    json.dumps(tickers),
                ),
            )
            if body:
                clean_imported_body(archive, article_id, body, title)
                body_count += 1
            imported += 1
        update_effective_dates(archive)
        archive.commit()
    return imported, body_count


def clean_prospective_archive(archive_path: Path) -> int:
    """Clean only newly fetched prospective bodies, avoiding a full reclean."""
    cleaned = 0
    with connect(archive_path) as archive:
        initialize_archive(archive)
        initialize_archive_provenance(archive)
        rows = archive.execute(
            """SELECT DISTINCT a.id,a.article_text_raw,
                              COALESCE(a.title_raw,a.title)
               FROM articles a JOIN prospective_article_sources p
                 ON p.article_id=a.id
               WHERE a.status='ok' AND a.article_text_raw IS NOT NULL
                 AND COALESCE(a.cleaning_version,'')<>?
               ORDER BY a.id""",
            (CLEANING_VERSION,),
        ).fetchall()
        for article_id, body, title in rows:
            clean_imported_body(archive, article_id, body, title)
            cleaned += 1
        update_effective_dates(archive)
        archive.commit()
    return cleaned


def print_status(database: Path) -> None:
    with connect(database) as db:
        initialize(db)
        articles = db.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
        tickers = db.execute(
            "SELECT COUNT(DISTINCT ticker) FROM article_tickers"
        ).fetchone()[0]
        runs = db.execute(
            "SELECT status,COUNT(*) FROM provider_runs GROUP BY status ORDER BY status"
        ).fetchall()
        latest = db.execute("SELECT MAX(last_retrieved_at) FROM articles").fetchone()[0]
    print(f"Prospective articles: {articles:,}")
    print(f"Tagged tickers: {tickers:,}")
    print(f"Latest retrieval: {latest or 'none'}")
    print("Runs: " + (", ".join(f"{status}={count}" for status, count in runs) or "none"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    commands = parser.add_subparsers(dest="command", required=True)
    alpha = commands.add_parser("sync-alpha")
    alpha.add_argument("--env-file", type=Path, default=Path(".env"))
    alpha.add_argument("--tickers", help="optional comma-separated provider ticker filter")
    alpha.add_argument("--time-from", help="Alpha timestamp YYYYMMDDTHHMM")
    alpha.add_argument("--hours-back", type=float, default=24.0)
    alpha.add_argument("--limit", type=int, default=1000)
    alpha.add_argument("--timeout", type=float, default=60.0)
    materialize_parser = commands.add_parser("materialize")
    materialize_parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    clean_parser = commands.add_parser(
        "clean-archive", help="clean newly fetched prospective bodies only"
    )
    clean_parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    commands.add_parser("status")
    args = parser.parse_args()
    if args.command == "sync-alpha":
        if not 1 <= args.limit <= 1000:
            parser.error("--limit must be between 1 and 1000")
        sync_alpha(args)
    elif args.command == "materialize":
        imported, bodies = materialize(args.database, args.archive)
        print(
            f"Materialized {imported:,} provider records into {args.archive}; "
            f"{bodies:,} provider bodies cleaned immediately and remaining URLs pending fetch",
            flush=True,
        )
    elif args.command == "clean-archive":
        cleaned = clean_prospective_archive(args.archive)
        print(
            f"Cleaned {cleaned:,} newly fetched prospective articles in "
            f"{args.archive}",
            flush=True,
        )
    else:
        print_status(args.database)


if __name__ == "__main__":
    main()
