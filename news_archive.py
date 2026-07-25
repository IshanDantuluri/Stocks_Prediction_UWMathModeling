#!/usr/bin/env python3
"""Migrate news events, fetch article text, and build a clean local corpus."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sqlite3
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

DEFAULT_CSV = Path("historical_news.csv")
DEFAULT_DB = Path("historical_news.sqlite3")
CLEANING_VERSION = "1.1.1"
EFFECTIVE_DATE_VERSION = "conservative-v1"
USER_AGENT = "Mozilla/5.0 (compatible; UWMathModelingNewsArchive/1.0)"
RETRYABLE = {408, 425, 429, 500, 502, 503, 504}
_local = threading.local()

DROP_SELECTORS = (
    "script", "style", "noscript", "svg", "nav", "footer", "form", "aside",
    "[role=navigation]", "[role=complementary]", "[aria-label*=advertisement i]",
    "[class*=advert i]", "[id*=advert i]", "[class*=newsletter i]",
    "[id*=newsletter i]", "[class*=subscribe i]", "[id*=subscribe i]",
    "[class*=related i]", "[id*=related i]", "[class*=recommend i]",
    "[class*=social i]", "[class*=share i]", "[class*=comment i]",
    "[id*=comment i]", "[class*=cookie i]", "[id*=cookie i]",
)
ARTICLE_SELECTORS = (
    "article", "[itemprop=articleBody]", "main", ".article-body", ".article__body",
    ".story-body", ".story__body", ".entry-content", ".post-content",
)
UI_PATTERNS = tuple(re.compile(p, re.I) for p in (
    r"^(sign up|subscribe)\b.*(newsletter|updates?|news)",
    r"^(already|not yet) (a )?subscriber\b",
    r"^(log in|sign in|register)( to| for)?\b",
    r"^(read|click|tap) (more|here)\b",
    r"^(share|print|save) (this|the) (article|story)\b",
    r"^follow (us|the .*?) on (facebook|twitter|x|instagram|youtube)\b",
    r"^follow .{1,100} on (facebook|twitter|x|instagram|youtube)\b",
    r"^(recommended|related|more) (stories|articles|content)\b",
    r"^comments? (have|must|are|will)\b",
    r"^(privacy policy|terms (of use|and conditions)|cookie settings)$",
    r"^get (the )?(editor'?s|latest|top|breaking)\b.*(newsletter|insights?|news)",
    r"^stay in the know\b",
    r"^klix\.ba čitajte i u našoj aplikaciji\b",
    r"^this article originally appeared on\b",
    r"^contributing:\s+",
    r"^photo\s*:\s*.+(?:getty|images?|ap|reuters)\b",
    r"^check all issues\s*&\s*supplements\.?$",
    r"^article continues below (this )?ad\.?$",
    r"^currently receiving \d+ of \d+ possible notifications\b",
    r"^sorry we are not currently accepting comments\b",
    r"^[_\-–—]{8,}$",
    r"^©.*all rights reserved\.?$",
))
TAIL_CUTOFF_PATTERNS = tuple(re.compile(p, re.I) for p in (
    r"^be the first to know when news breaks\.?$",
    r"^today'?s top stories curated by our news team\.?$",
    r"^your digital replica of today'?s paper\b",
    r"^sharp\. close to the ground\. digging deep\.",
    r"^every saturday and tuesday, explore destinations\b",
    r"^your essential national news digest\b",
    r"^get news, reviews and expert insights every thursday\b",
    r"^real local, smart property news\b",
    r"^test your skills with interactive crosswords\b",
    r"^temi più discussi\s*:",
))
CREDIT_PATTERN = re.compile(
    r"^(australian associated press|associated press|reuters|afp|copyright .+|"
    r".{2,100},\s*the associated press)$", re.I
)
BYLINE_PATTERN = re.compile(r"^(?:by\s+)[\w .,'’\-]{2,100}$", re.I)
AUTHOR_BIO_PATTERN = re.compile(
    r"\b(is (a|an) (writer|reporter|editor)|follow (him|her|them) on|contributed to this report)\b",
    re.I,
)
GENERIC_TITLE_PATTERN = re.compile(
    r"^(home|homepage|latest news|breaking news|global news|news|welcome)(\s*[|\-].*)?$", re.I
)
WORD_RE = re.compile(r"\b\w+\b", re.UNICODE)


def connect(path: Path) -> sqlite3.Connection:
    db = sqlite3.connect(path)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA foreign_keys=ON")
    return db


def _add_missing_columns(db: sqlite3.Connection) -> None:
    columns = {row[1] for row in db.execute("PRAGMA table_info(articles)")}
    additions = {
        "article_text_raw": "TEXT", "article_text_clean": "TEXT", "title_raw": "TEXT",
        "author": "TEXT", "published_at": "TEXT", "language": "TEXT",
        "cleaning_version": "TEXT", "word_count": "INTEGER",
        "quality_score": "REAL", "quality_status": "TEXT",
        "quality_reasons": "TEXT", "content_hash": "TEXT",
        "simhash": "TEXT", "canonical_article_id": "INTEGER",
        "effective_date": "TEXT", "effective_date_source": "TEXT",
        "effective_date_version": "TEXT",
    }
    for name, sql_type in additions.items():
        if name not in columns:
            db.execute(f"ALTER TABLE articles ADD COLUMN {name} {sql_type}")
    # Preserve text collected by the original fetcher.
    db.execute(
        "UPDATE articles SET article_text_raw=article_text "
        "WHERE article_text_raw IS NULL AND article_text IS NOT NULL"
    )
    db.execute("UPDATE articles SET title_raw=title WHERE title_raw IS NULL AND title IS NOT NULL")


def initialize(db: sqlite3.Connection) -> None:
    db.executescript(
        """
        CREATE TABLE IF NOT EXISTS articles (
            id INTEGER PRIMARY KEY,
            source_url TEXT NOT NULL UNIQUE,
            final_url TEXT,
            domain TEXT,
            title TEXT,
            article_text TEXT,
            status TEXT NOT NULL DEFAULT 'pending'
                CHECK(status IN ('pending', 'fetching', 'ok', 'empty', 'http_error', 'error')),
            http_status INTEGER,
            fetched_at TEXT,
            error TEXT
        );
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY,
            date TEXT NOT NULL,
            event_category TEXT,
            primary_actor TEXT,
            location TEXT,
            goldstein_impact REAL,
            article_sentiment REAL,
            media_coverage_volume INTEGER,
            article_id INTEGER NOT NULL REFERENCES articles(id)
        );
        CREATE TABLE IF NOT EXISTS article_fingerprint_bands (
            band INTEGER NOT NULL,
            bucket INTEGER NOT NULL,
            article_id INTEGER NOT NULL REFERENCES articles(id),
            PRIMARY KEY (band, bucket, article_id)
        );
        CREATE INDEX IF NOT EXISTS events_date_idx ON events(date);
        CREATE INDEX IF NOT EXISTS events_article_idx ON events(article_id);
        CREATE INDEX IF NOT EXISTS articles_status_idx ON articles(status);
        CREATE INDEX IF NOT EXISTS fingerprint_lookup_idx
            ON article_fingerprint_bands(band, bucket);
        """
    )
    _add_missing_columns(db)
    db.execute("CREATE INDEX IF NOT EXISTS articles_quality_idx ON articles(quality_status)")
    db.execute("CREATE INDEX IF NOT EXISTS articles_hash_idx ON articles(content_hash)")
    db.commit()


def number(value: str, kind):
    try:
        return kind(value)
    except (TypeError, ValueError):
        return None


def normalize_date(value: str) -> str:
    for fmt in ("%m/%d/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt).date().isoformat()
        except ValueError:
            pass
    return value


def migrate_csv(db: sqlite3.Connection, csv_path: Path) -> tuple[int, int]:
    existing = db.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    if existing:
        articles = db.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
        return existing, articles
    with csv_path.open(newline="", encoding="utf-8-sig") as source:
        for i, row in enumerate(csv.DictReader(source), 1):
            url = row["source_url"].strip()
            domain = urlparse(url).netloc.lower().removeprefix("www.")
            db.execute("INSERT OR IGNORE INTO articles(source_url, domain) VALUES (?, ?)", (url, domain))
            article_id = db.execute("SELECT id FROM articles WHERE source_url=?", (url,)).fetchone()[0]
            db.execute(
                """INSERT INTO events(date,event_category,primary_actor,location,
                   goldstein_impact,article_sentiment,media_coverage_volume,article_id)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (normalize_date(row["date"]), row["event_category"], row["primary_actor"],
                 row["location"], number(row["goldstein_impact"], float),
                 number(row["article_sentiment"], float),
                 number(row["media_coverage_volume"], int), article_id),
            )
            if i % 5_000 == 0:
                db.commit()
                print(f"Migrated {i:,} events...", flush=True)
    db.commit()
    return db.execute("SELECT COUNT(*) FROM events").fetchone()[0], db.execute("SELECT COUNT(*) FROM articles").fetchone()[0]


def _meta(soup: BeautifulSoup, *keys: str) -> str | None:
    for key in keys:
        tag = soup.find("meta", attrs={"property": key}) or soup.find("meta", attrs={"name": key})
        if tag and tag.get("content"):
            return " ".join(tag["content"].split())
    return None


def _paragraphs(root) -> list[str]:
    result = []
    for p in root.find_all("p"):
        text = " ".join(p.get_text(" ", strip=True).split())
        if len(text) >= 25:
            result.append(text)
    return result


def extract_article(html: bytes) -> dict:
    soup = BeautifulSoup(html, "html.parser")
    title = _meta(soup, "og:title", "twitter:title")
    if not title and soup.title and soup.title.string:
        title = " ".join(soup.title.string.split())
    author = _meta(soup, "author", "article:author", "byl")
    published = _meta(soup, "article:published_time", "datePublished", "date")
    language = soup.html.get("lang") if soup.html else None
    for selector in DROP_SELECTORS:
        for tag in soup.select(selector):
            tag.decompose()
    candidates = []
    seen = set()
    for selector in ARTICLE_SELECTORS:
        for root in soup.select(selector):
            if id(root) not in seen:
                paragraphs = _paragraphs(root)
                candidates.append((sum(map(len, paragraphs)), paragraphs))
                seen.add(id(root))
    if not candidates:
        root = soup.body or soup
        paragraphs = _paragraphs(root)
    else:
        paragraphs = max(candidates, key=lambda item: item[0])[1]
    raw = "\n\n".join(paragraphs).strip() or None
    return {"text": raw, "title": title, "author": author,
            "published_at": published, "language": language}


# Backward-compatible helper used by early callers and tests.
def extract_text(html: bytes) -> tuple[str | None, str]:
    result = extract_article(html)
    return result["text"], result["title"] or ""


def session() -> requests.Session:
    if not hasattr(_local, "session"):
        _local.session = requests.Session()
        _local.session.headers.update({"User-Agent": USER_AGENT, "Accept": "text/html"})
    return _local.session


def fetch(article_id: int, url: str, timeout: int) -> dict:
    result = {"id": article_id, "status": "error", "http_status": None,
              "final_url": None, "error": None, "text": None, "title": None,
              "author": None, "published_at": None, "language": None}
    try:
        response = session().get(url, timeout=timeout, allow_redirects=True)
        result.update(http_status=response.status_code, final_url=response.url)
        if response.status_code != 200:
            result["status"] = "pending" if response.status_code in RETRYABLE else "http_error"
            result["error"] = f"HTTP {response.status_code}"
            return result
        content_type = response.headers.get("Content-Type", "").lower()
        if "html" not in content_type:
            result.update(status="empty", error=f"Unsupported content type: {content_type}")
            return result
        extracted = extract_article(response.content)
        result.update(extracted)
        result["status"] = "ok" if extracted["text"] else "empty"
        return result
    except requests.RequestException as exc:
        result["error"] = str(exc)[:500]
        return result


def download(db: sqlite3.Connection, limit: int | None, workers: int, timeout: int,
             retry_errors: bool, retry_http_errors: bool, newest_first: bool) -> None:
    if retry_errors:
        db.execute("UPDATE articles SET status='pending' WHERE status='error'")
    if retry_http_errors:
        db.execute("UPDATE articles SET status='pending' WHERE status='http_error'")
    if retry_errors or retry_http_errors:
        db.commit()
    order = "DESC" if newest_first else "ASC"
    sql = f"SELECT id,source_url FROM articles WHERE status='pending' ORDER BY id {order}"
    params: tuple = ()
    if limit is not None:
        sql += " LIMIT ?"
        params = (limit,)
    pending = db.execute(sql, params).fetchall()
    print(f"Fetching {len(pending):,} unique URLs with {workers} workers...", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fetch, article_id, url, timeout) for article_id, url in pending]
        for completed, future in enumerate(as_completed(futures), 1):
            item = future.result()
            db.execute(
                """UPDATE articles SET status=?,http_status=?,final_url=?,title=?,title_raw=?,author=?,
                   published_at=?,language=?,article_text=?,article_text_raw=?,error=?,fetched_at=?,
                   article_text_clean=NULL,cleaning_version=NULL,quality_status=NULL
                   WHERE id=?""",
                (item["status"], item["http_status"], item["final_url"], item["title"], item["title"],
                 item["author"], item["published_at"], item["language"], item["text"],
                 item["text"], item["error"], datetime.now(timezone.utc).isoformat(), item["id"]),
            )
            if completed % 100 == 0:
                db.commit()
                print(f"Fetched {completed:,}/{len(pending):,}", flush=True)
    db.commit()


def normalize_paragraph(text: str) -> str:
    text = re.sub(r"https?://\S+", "", text.lower())
    text = re.sub(r"\d+", "#", text)
    return " ".join(WORD_RE.findall(text))


def learn_boilerplate(db: sqlite3.Connection) -> dict[str, set[str]]:
    counts: dict[str, Counter] = defaultdict(Counter)
    totals = Counter()
    for domain, raw in db.execute(
        "SELECT domain,article_text_raw FROM articles WHERE status='ok' AND article_text_raw IS NOT NULL"
    ):
        totals[domain] += 1
        paragraphs = [p for p in raw.split("\n\n") if 40 <= len(p) <= 500]
        # Repeated boilerplate is overwhelmingly at page edges.
        edge = paragraphs[:2] + paragraphs[-3:]
        counts[domain].update(set(normalize_paragraph(p) for p in edge))
    learned = {}
    for domain, counter in counts.items():
        threshold = max(5, math.ceil(totals[domain] * 0.15))
        learned[domain] = {p for p, count in counter.items() if p and count >= threshold}
    return learned


def clean_title(title: str | None) -> str | None:
    if not title:
        return None
    title = " ".join(title.split())
    if " | " in title:
        title = title.split(" | ", 1)[0].strip()
    return title or None


def clean_article(raw: str | None, title: str | None, boilerplate: set[str] | None = None) -> dict:
    boilerplate = boilerplate or set()
    reasons = []
    kept = []
    removed = 0
    paragraphs = [" ".join(p.split()).strip() for p in (raw or "").split("\n\n") if p.strip()]
    # Certain publishers append a list of newsletters or related headlines. A
    # recognized marker near the tail is safer than deleting phrases globally.
    cutoff = None
    for index, paragraph in enumerate(paragraphs):
        if index >= max(2, int(len(paragraphs) * 0.4)) and any(
            pattern.search(paragraph) for pattern in TAIL_CUTOFF_PATTERNS
        ):
            cutoff = index
            break
    if cutoff is not None:
        removed += len(paragraphs) - cutoff
        paragraphs = paragraphs[:cutoff]
        reasons.append("promotional_tail_removed")

    for index, paragraph in enumerate(paragraphs):
        paragraph = " ".join(paragraph.split()).strip()
        if not paragraph:
            continue
        normalized = normalize_paragraph(paragraph)
        is_edge = index < 2 or index >= max(0, len(paragraphs) - 3)
        noisy = any(pattern.search(paragraph) for pattern in UI_PATTERNS)
        noisy = noisy or normalized in boilerplate
        noisy = noisy or (is_edge and AUTHOR_BIO_PATTERN.search(paragraph) is not None)
        noisy = noisy or (index == 0 and BYLINE_PATTERN.fullmatch(paragraph) is not None)
        noisy = noisy or (is_edge and CREDIT_PATTERN.fullmatch(paragraph) is not None)
        if noisy:
            removed += 1
        else:
            kept.append(paragraph)

    # Collapse exact duplicates and cumulative DOM fragments while retaining the
    # longest version. Restrict containment checks to adjacent paragraphs.
    collapsed = []
    normalized_seen = set()
    duplicate_removed = 0
    for paragraph in kept:
        normalized = normalize_paragraph(paragraph)
        if normalized in normalized_seen:
            duplicate_removed += 1
            continue
        if collapsed:
            previous = normalize_paragraph(collapsed[-1])
            if len(previous) >= 60 and previous in normalized and len(previous) / len(normalized) >= 0.35:
                normalized_seen.discard(previous)
                collapsed[-1] = paragraph
                normalized_seen.add(normalized)
                duplicate_removed += 1
                continue
            if len(normalized) >= 60 and normalized in previous and len(normalized) / len(previous) >= 0.35:
                duplicate_removed += 1
                continue
        collapsed.append(paragraph)
        normalized_seen.add(normalized)
    if duplicate_removed:
        reasons.append("duplicate_fragments_removed")
        removed += duplicate_removed
    kept = collapsed
    clean = "\n\n".join(kept).strip() or None
    words = len(WORD_RE.findall(clean or ""))
    paragraphs = len(kept)
    cleaned_title = clean_title(title)
    score = 1.0
    if words < 40:
        reasons.append("too_short")
        score -= 0.75
    elif words < 100:
        reasons.append("short_article")
        score -= 0.25
    if paragraphs <= 1:
        reasons.append("single_paragraph")
        score -= 0.15
    if not cleaned_title or GENERIC_TITLE_PATTERN.match(cleaned_title):
        reasons.append("missing_or_generic_title")
        score -= 0.25
    total = paragraphs + removed
    if total and removed / total > 0.4:
        reasons.append("mostly_boilerplate")
        score -= 0.3
    score = max(0.0, min(1.0, score))
    if words < 40 or score < 0.35:
        status = "rejected"
    elif words < 100 or score < 0.7:
        status = "questionable"
    else:
        status = "usable"
    return {"text": clean, "title": cleaned_title, "word_count": words,
            "score": score, "status": status, "reasons": reasons}


def content_hash(text: str) -> str:
    normalized = " ".join(WORD_RE.findall(text.lower()))
    return hashlib.sha256(normalized.encode()).hexdigest()


def simhash(text: str) -> int:
    words = WORD_RE.findall(text.lower())
    features = (" ".join(words[i:i + 4]) for i in range(max(1, len(words) - 3)))
    vector = [0] * 64
    for feature in features:
        value = int.from_bytes(hashlib.blake2b(feature.encode(), digest_size=8).digest(), "big")
        for bit in range(64):
            vector[bit] += 1 if value & (1 << bit) else -1
    return sum((1 << bit) for bit, weight in enumerate(vector) if weight >= 0)


def _near_duplicate(db: sqlite3.Connection, fingerprint: int, word_count: int) -> int | None:
    candidates = set()
    for band in range(4):
        bucket = (fingerprint >> (band * 16)) & 0xFFFF
        candidates.update(row[0] for row in db.execute(
            "SELECT article_id FROM article_fingerprint_bands WHERE band=? AND bucket=?",
            (band, bucket),
        ))
    for candidate in sorted(candidates):
        row = db.execute(
            "SELECT simhash,word_count,canonical_article_id FROM articles "
            "WHERE id=? AND quality_status='usable'", (candidate,)
        ).fetchone()
        if not row or not row[0] or not row[1]:
            continue
        if min(word_count, row[1]) / max(word_count, row[1]) < 0.85:
            continue
        if (fingerprint ^ int(row[0], 16)).bit_count() <= 3:
            return row[2] or candidate
    return None


def reject_domain_fallback_pages(db: sqlite3.Connection) -> int:
    """Reject a common failure mode: many dead URLs returning one domain homepage."""
    groups = db.execute(
        """SELECT domain,content_hash,COUNT(*) FROM articles
           WHERE content_hash IS NOT NULL GROUP BY domain,content_hash HAVING COUNT(*) >= 3"""
    ).fetchall()
    rejected = 0
    for domain, digest, _ in groups:
        rows = db.execute(
            "SELECT id,quality_reasons FROM articles WHERE domain=? AND content_hash=?",
            (domain, digest),
        ).fetchall()
        for article_id, reasons_json in rows:
            reasons = json.loads(reasons_json or "[]")
            if "repeated_domain_fallback" not in reasons:
                reasons.append("repeated_domain_fallback")
            db.execute(
                "UPDATE articles SET quality_status='rejected',quality_score=0.0,quality_reasons=? WHERE id=?",
                (json.dumps(reasons), article_id),
            )
            rejected += 1
    return rejected


def clean_corpus(
    db: sqlite3.Connection,
    reject_repeated_domain_fallback: bool = True,
) -> None:
    learned = learn_boilerplate(db)
    rows = db.execute(
        "SELECT id,domain,COALESCE(title_raw,title),article_text_raw FROM articles "
        "WHERE status='ok' AND article_text_raw IS NOT NULL ORDER BY id"
    ).fetchall()
    db.execute("DELETE FROM article_fingerprint_bands")
    print(f"Cleaning {len(rows):,} downloaded articles (version {CLEANING_VERSION})...", flush=True)
    for completed, (article_id, domain, title, raw) in enumerate(rows, 1):
        item = clean_article(raw, title, learned.get(domain))
        digest = content_hash(item["text"]) if item["text"] else None
        fingerprint = simhash(item["text"]) if item["text"] else None
        canonical = None
        if digest:
            exact = db.execute(
                "SELECT COALESCE(canonical_article_id,id) FROM articles "
                "WHERE content_hash=? AND id<? ORDER BY id LIMIT 1", (digest, article_id)
            ).fetchone()
            canonical = exact[0] if exact else (
                _near_duplicate(db, fingerprint, item["word_count"])
                if item["status"] == "usable" and item["word_count"] >= 100 else None
            )
        canonical = canonical or article_id
        db.execute(
            """UPDATE articles SET title=?,article_text_clean=?,cleaning_version=?,word_count=?,
               quality_score=?,quality_status=?,quality_reasons=?,content_hash=?,simhash=?,
               canonical_article_id=? WHERE id=?""",
            (item["title"], item["text"], CLEANING_VERSION, item["word_count"],
             item["score"], item["status"], json.dumps(item["reasons"]), digest,
             f"{fingerprint:016x}" if fingerprint is not None else None, canonical, article_id),
        )
        if fingerprint is not None:
            for band in range(4):
                bucket = (fingerprint >> (band * 16)) & 0xFFFF
                db.execute("INSERT INTO article_fingerprint_bands VALUES (?,?,?)", (band, bucket, article_id))
        if completed % 1_000 == 0:
            db.commit()
            print(f"Cleaned {completed:,}/{len(rows):,}", flush=True)
    if reject_repeated_domain_fallback:
        fallback_count = reject_domain_fallback_pages(db)
        if fallback_count:
            print(f"Rejected {fallback_count:,} repeated domain fallback pages", flush=True)
    db.commit()


def _parse_date_prefix(value: str | None):
    if not value or len(value) < 10:
        return None
    try:
        return datetime.strptime(value[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def update_effective_dates(db: sqlite3.Connection) -> dict[str, int]:
    """Choose a conservative availability date without ever moving news earlier."""
    _add_missing_columns(db)
    event_dates = {}
    for root_id, value in db.execute(
        """SELECT COALESCE(a.canonical_article_id,a.id),MIN(e.date)
           FROM events e JOIN articles a ON a.id=e.article_id
           GROUP BY COALESCE(a.canonical_article_id,a.id)"""
    ):
        parsed = _parse_date_prefix(value)
        if parsed:
            event_dates[root_id] = parsed
    published_dates: dict[int, list] = defaultdict(list)
    for article_id, root_id, value in db.execute(
        """SELECT id,COALESCE(canonical_article_id,id),published_at
           FROM articles WHERE published_at IS NOT NULL"""
    ):
        parsed = _parse_date_prefix(value)
        if parsed:
            published_dates[root_id].append(parsed)
    root_values = {}
    stats = Counter()
    for root_id, event_date in event_dates.items():
        candidates = published_dates.get(root_id, [])
        published_date = min(candidates) if candidates else None
        if published_date and published_date > event_date:
            effective = published_date
            source = "published_at_later"
            stats["moved_later"] += 1
        elif published_date:
            effective = event_date
            source = "event_date_published_not_later"
            stats["published_not_later"] += 1
        else:
            effective = event_date
            source = "event_date_only"
            stats["event_only"] += 1
        root_values[root_id] = (
            effective.isoformat(),
            source,
            EFFECTIVE_DATE_VERSION,
        )
    updates = []
    for article_id, root_id in db.execute(
        "SELECT id,COALESCE(canonical_article_id,id) FROM articles"
    ):
        value = root_values.get(root_id)
        if value:
            updates.append((*value, article_id))
    db.executemany(
        """UPDATE articles SET effective_date=?,effective_date_source=?,
               effective_date_version=? WHERE id=?""",
        updates,
    )
    db.commit()
    stats["articles_updated"] = len(updates)
    stats["canonical_dates"] = len(root_values)
    return dict(stats)


def print_summary(db: sqlite3.Connection, db_path: Path) -> None:
    print("\nArchive summary")
    print(f"  Database: {db_path} ({db_path.stat().st_size / 1024**2:.1f} MiB)")
    print(f"  Events: {db.execute('SELECT COUNT(*) FROM events').fetchone()[0]:,}")
    print(f"  Unique articles: {db.execute('SELECT COUNT(*) FROM articles').fetchone()[0]:,}")
    for status, count in db.execute("SELECT status,COUNT(*) FROM articles GROUP BY status ORDER BY status"):
        print(f"  fetch {status}: {count:,}")
    for status, count in db.execute(
        "SELECT quality_status,COUNT(*) FROM articles WHERE quality_status IS NOT NULL GROUP BY quality_status ORDER BY quality_status"
    ):
        print(f"  quality {status}: {count:,}")
    duplicates = db.execute("SELECT COUNT(*) FROM articles WHERE canonical_article_id IS NOT NULL AND canonical_article_id != id").fetchone()[0]
    print(f"  duplicate articles: {duplicates:,}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--migrate-only", action="store_true")
    parser.add_argument("--clean-only", action="store_true", help="clean fetched text without downloading")
    parser.add_argument(
        "--dates-only",
        action="store_true",
        help="recompute conservative effective dates without fetching or cleaning",
    )
    parser.add_argument("--no-clean", action="store_true", help="do not clean after downloading")
    parser.add_argument("--limit", type=int, help="maximum URLs to fetch this run")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument("--newest-first", action="store_true")
    parser.add_argument(
        "--retry-errors", action="store_true",
        help="retry network errors such as DNS failures and timeouts",
    )
    parser.add_argument(
        "--retry-http-errors", action="store_true",
        help="also retry non-retryable HTTP responses such as 403, 404, and 410",
    )
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 16:
        parser.error("--workers must be between 1 and 16")
    started = time.monotonic()
    with connect(args.db) as db:
        initialize(db)
        if args.dates_only:
            stats = update_effective_dates(db)
            print(
                "Effective dates updated: "
                f"{stats.get('canonical_dates', 0):,} canonical articles | "
                f"{stats.get('moved_later', 0):,} moved later | "
                f"{stats.get('event_only', 0):,} without publication metadata"
            )
            print_summary(db, args.db)
            return
        events, articles = migrate_csv(db, args.csv)
        print(f"Migration complete: {events:,} events, {articles:,} unique URLs")
        if not args.migrate_only and not args.clean_only:
            download(
                db, args.limit, args.workers, args.timeout, args.retry_errors,
                args.retry_http_errors, args.newest_first,
            )
        if not args.migrate_only and not args.no_clean:
            clean_corpus(db)
        update_effective_dates(db)
        print_summary(db, args.db)
    print(f"Finished in {time.monotonic() - started:.1f}s")


if __name__ == "__main__":
    main()
