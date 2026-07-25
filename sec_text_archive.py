#!/usr/bin/env python3
"""Resumable SEC 8-K/6-K text ingestion for the article embedding pipeline.

The archive is intentionally separate from historical_news.sqlite3.  It uses
the same ``articles`` schema, cleaning functions, and chunker, while retaining
SEC-specific point-in-time metadata and compressed source bytes.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import gzip
import hashlib
import json
import os
import re
import sqlite3
import subprocess
import tempfile
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Iterable
from urllib.parse import quote

import requests
from bs4 import BeautifulSoup

from news_archive import (
    clean_corpus,
    connect as connect_articles,
    extract_article,
    initialize as initialize_articles,
    update_effective_dates,
)

DEFAULT_SOURCE_DB = Path("point_in_time_data.sqlite3")
DEFAULT_ARCHIVE_DB = Path("sec_text_archive.sqlite3")
DEFAULT_RAW_DIR = Path("sec_text_raw")
FORMS = ("8-K", "8-K/A", "6-K", "6-K/A")
RETRYABLE_HTTP = {408, 425, 429, 500, 502, 503, 504}
TEXT_SUFFIXES = {".htm", ".html", ".txt"}
EXHIBIT_99_RE = re.compile(r"^(?:EX[-\s]?)?99(?:\.\d+)?$", re.I)
PRESS_RELEASE_RE = re.compile(r"\b(press release|news release|earnings release)\b", re.I)
LARGE_SEC_HTML_BYTES = 1_000_000
MAX_SEC_TEXT_CHARS = 50_000
TRUNCATION_NOTICE = (
    f"\n\n[SEC TEXT TRUNCATED AFTER {MAX_SEC_TEXT_CHARS:,} CHARACTERS; "
    "THE COMPLETE RAW FILING IS RETAINED.]"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds + 0.5))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


def rate_and_eta(completed: int, total: int, started: float, unit: str) -> str:
    elapsed = max(time.monotonic() - started, 1e-6)
    rate = completed / elapsed
    eta = (total - completed) / rate if rate > 0 else 0
    return f"{rate:.2f} {unit}/s | ETA {format_duration(eta)}"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def accession_compact(accession: str) -> str:
    return accession.replace("-", "")


def archive_base_url(cik: int, accession: str) -> str:
    return (
        f"https://www.sec.gov/Archives/edgar/data/{cik}/"
        f"{accession_compact(accession)}"
    )


def filing_index_url(cik: int, accession: str) -> str:
    return f"{archive_base_url(cik, accession)}/{accession}-index.html"


def document_url(cik: int, accession: str, document_name: str) -> str:
    # SEC filenames are path components.  Quoting prevents accidental query or
    # fragment interpretation while retaining common punctuation.
    return f"{archive_base_url(cik, accession)}/{quote(Path(document_name).name)}"


def default_sec_user_agent() -> str | None:
    configured = os.environ.get("SEC_USER_AGENT", "").strip()
    if configured:
        return configured
    name = subprocess.run(
        ["git", "config", "user.name"], capture_output=True, text=True, check=False
    ).stdout.strip()
    email = subprocess.run(
        ["git", "config", "user.email"], capture_output=True, text=True, check=False
    ).stdout.strip()
    return f"{name} {email}" if name and email else None


class RateLimiter:
    def __init__(self, requests_per_second: float) -> None:
        if not 0 < requests_per_second <= 10:
            raise ValueError("SEC request rate must be >0 and <=10 requests/second")
        self.minimum_interval = 1.0 / requests_per_second
        self.last_request = 0.0
        self.request_count = 0
        self.lock = threading.Lock()

    def wait(self) -> None:
        # Serialize request starts across all worker threads.  Responses,
        # extraction, compression, and database preparation may overlap, but
        # SEC request starts remain globally capped.
        with self.lock:
            delay = self.minimum_interval - (time.monotonic() - self.last_request)
            if delay > 0:
                time.sleep(delay)
            self.last_request = time.monotonic()
            self.request_count += 1

    def count(self) -> int:
        with self.lock:
            return self.request_count


class SecClient:
    def __init__(self, user_agent: str, requests_per_second: float, retries: int) -> None:
        self.headers = {
            "User-Agent": user_agent,
            "Accept-Encoding": "gzip, deflate",
            "Accept": "text/html,text/plain;q=0.9,*/*;q=0.1",
        }
        self.local = threading.local()
        self.limiter = RateLimiter(requests_per_second)
        self.retries = retries
        self.cancelled = threading.Event()

    def cancel(self) -> None:
        self.cancelled.set()

    def session(self) -> requests.Session:
        session = getattr(self.local, "session", None)
        if session is None:
            session = requests.Session()
            session.headers.update(self.headers)
            self.local.session = session
        return session

    def get(self, url: str) -> tuple[bytes, int, str]:
        last_error = ""
        for attempt in range(self.retries + 1):
            if self.cancelled.is_set():
                return b"", 0, "cancelled"
            self.limiter.wait()
            try:
                # Separate connection and read timeouts prevent one unhealthy
                # archive edge from pinning a worker for 90 seconds.
                response = self.session().get(url, timeout=(10, 30))
                if response.status_code in RETRYABLE_HTTP:
                    last_error = f"HTTP {response.status_code}"
                elif response.status_code >= 400:
                    return response.content, response.status_code, f"HTTP {response.status_code}"
                else:
                    return response.content, response.status_code, ""
            except requests.RequestException as exc:
                last_error = f"{type(exc).__name__}: {str(exc)[:300]}"
            if self.cancelled.is_set():
                return b"", 0, "cancelled"
            if attempt < self.retries:
                if self.cancelled.wait(min(30, 2 ** attempt)):
                    return b"", 0, "cancelled"
        return b"", 0, last_error or "request failed"


def completion_order_map(
    function,
    rows: list[tuple[object, ...]],
    workers: int,
    cancel=None,
) -> Iterable[tuple[object, ...]]:
    """Run bounded work and yield whichever result completes first.

    ThreadPoolExecutor.map yields in input order and eagerly queues the entire
    corpus.  A single slow first SEC response can therefore make a healthy run
    look frozen.  Keeping at most two tasks per worker in flight also makes
    Ctrl-C cancellation bounded and avoids retaining thousands of futures.
    """
    if workers <= 1:
        yield from map(function, rows)
        return

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=workers)
    source = iter(rows)
    pending: set[concurrent.futures.Future] = set()
    try:
        for _ in range(min(len(rows), workers * 2)):
            pending.add(executor.submit(function, next(source)))
        while pending:
            done, pending = concurrent.futures.wait(
                pending, return_when=concurrent.futures.FIRST_COMPLETED
            )
            for future in done:
                yield future.result()
                try:
                    row = next(source)
                except StopIteration:
                    continue
                pending.add(executor.submit(function, row))
    except BaseException:
        if cancel is not None:
            cancel()
        for future in pending:
            future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)


def staged_completion_order_map(
    download,
    process,
    rows: list[tuple[object, ...]],
    download_workers: int,
    process_workers: int,
    cancel=None,
    process_pool: bool = False,
) -> Iterable[tuple[object, ...]]:
    """Keep SEC downloads full while bounded processing runs separately.

    Combining HTTP, gzip writes, and HTML extraction in one worker pool makes
    processing time reduce request concurrency.  This two-stage scheduler keeps
    a bounded number of response bodies in memory and yields completed processed
    results in completion order.
    """
    if download_workers <= 0 or process_workers <= 0:
        raise ValueError("download and processing worker counts must be positive")

    download_executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=download_workers,
        thread_name_prefix="sec-download",
    )
    if process_pool:
        # BeautifulSoup extraction is GIL-heavy for large SEC filings.  A
        # process pool allows those documents to use multiple CPU cores.
        process_executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=process_workers,
        )
    else:
        process_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=process_workers,
            thread_name_prefix="sec-process",
        )
    source = iter(rows)
    pending_downloads: set[concurrent.futures.Future] = set()
    pending_processing: set[concurrent.futures.Future] = set()
    downloaded = deque()
    download_target = min(len(rows), download_workers * 2)
    processing_target = process_workers * 2
    # Prevent large filings from accumulating without bound if extraction falls
    # behind.  The allowance still provides enough slack to keep HTTP workers
    # occupied through short processing bursts.
    processing_backlog_limit = max(processing_target, download_workers * 2)

    def refill_downloads() -> None:
        while (
            len(pending_downloads) < download_target
            and len(downloaded) + len(pending_processing) < processing_backlog_limit
        ):
            try:
                row = next(source)
            except StopIteration:
                break
            pending_downloads.add(download_executor.submit(download, row))

    try:
        refill_downloads()
        while pending_downloads or pending_processing or downloaded:
            while downloaded and len(pending_processing) < processing_target:
                pending_processing.add(
                    process_executor.submit(process, downloaded.popleft())
                )
            refill_downloads()

            all_pending = pending_downloads | pending_processing
            if not all_pending:
                continue
            done, _ = concurrent.futures.wait(
                all_pending, return_when=concurrent.futures.FIRST_COMPLETED
            )
            for future in done:
                if future in pending_downloads:
                    pending_downloads.remove(future)
                    downloaded.append(future.result())
                else:
                    pending_processing.remove(future)
                    yield future.result()
    except BaseException:
        if cancel is not None:
            cancel()
        for future in pending_downloads | pending_processing:
            future.cancel()
        download_executor.shutdown(wait=False, cancel_futures=True)
        process_executor.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        download_executor.shutdown(wait=True)
        process_executor.shutdown(wait=True)


def request_rate(
    client: object, initial_count: int | None, started: float
) -> str:
    limiter = getattr(client, "limiter", None)
    if initial_count is None or limiter is None:
        return ""
    elapsed = max(time.monotonic() - started, 1e-6)
    return f" | {(limiter.count() - initial_count) / elapsed:.2f} requests/s"


@dataclass(frozen=True)
class DocumentChoice:
    sequence: int | None
    document_name: str
    description: str
    exhibit_type: str
    size_bytes: int | None
    is_primary: bool
    selection_reason: str


def parse_size(value: str) -> int | None:
    digits = re.sub(r"\D", "", value or "")
    return int(digits) if digits else None


def normalize_exhibit_type(value: str) -> str:
    return re.sub(r"\s+", "", (value or "").upper())


def is_text_document(name: str) -> bool:
    return Path(name.lower()).suffix in TEXT_SUFFIXES


def choose_documents(index_html: bytes, primary_document: str | None) -> list[DocumentChoice]:
    """Select the primary filing and textual Exhibit 99.x documents."""
    soup = BeautifulSoup(index_html, "html.parser")
    candidates: list[DocumentChoice] = []
    seen: set[str] = set()
    primary_name = Path(primary_document or "").name.lower()
    for table in soup.select("table.tableFile, table[summary*='Document Format Files' i]"):
        for row in table.select("tr"):
            cells = row.find_all("td")
            if len(cells) < 4:
                continue
            link = cells[2].find("a")
            name = Path((link.get_text(" ", strip=True) if link else cells[2].get_text(" ", strip=True))).name
            if not name or name.lower() in seen or not is_text_document(name):
                continue
            seen.add(name.lower())
            sequence_text = cells[0].get_text(" ", strip=True)
            sequence = int(sequence_text) if sequence_text.isdigit() else None
            description = cells[1].get_text(" ", strip=True)
            exhibit_type = normalize_exhibit_type(cells[3].get_text(" ", strip=True))
            size = parse_size(cells[4].get_text(" ", strip=True)) if len(cells) > 4 else None
            is_primary = bool(primary_name and name.lower() == primary_name)
            is_exhibit_99 = EXHIBIT_99_RE.fullmatch(exhibit_type) is not None
            is_press_release = PRESS_RELEASE_RE.search(description) is not None
            if is_primary:
                reason = "primary_document"
            elif is_exhibit_99:
                reason = "exhibit_99"
            elif is_press_release:
                reason = "press_release_description"
            else:
                continue
            candidates.append(
                DocumentChoice(
                    sequence,
                    name,
                    description,
                    exhibit_type,
                    size,
                    is_primary,
                    reason,
                )
            )
    return sorted(
        candidates,
        key=lambda item: (
            0 if item.is_primary else 1,
            item.sequence if item.sequence is not None else 1_000_000,
            item.document_name,
        ),
    )


def initialize(db: sqlite3.Connection) -> None:
    initialize_articles(db)
    db.executescript(
        """
        CREATE TABLE IF NOT EXISTS sec_ingestion_runs (
            run_id INTEGER PRIMARY KEY,
            started_at TEXT NOT NULL,
            completed_at TEXT,
            command TEXT NOT NULL,
            status TEXT NOT NULL,
            filings_discovered INTEGER NOT NULL DEFAULT 0,
            documents_fetched INTEGER NOT NULL DEFAULT 0,
            documents_failed INTEGER NOT NULL DEFAULT 0,
            error TEXT
        );

        CREATE TABLE IF NOT EXISTS sec_filing_queue (
            accession TEXT PRIMARY KEY,
            cik INTEGER NOT NULL,
            filing_date TEXT NOT NULL,
            accepted_at TEXT NOT NULL,
            available_at_quality TEXT NOT NULL,
            form TEXT NOT NULL,
            primary_document TEXT,
            source_metadata_retrieved_at TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending'
                CHECK(status IN ('pending','discovered','complete','no_documents','http_error','error')),
            index_url TEXT NOT NULL,
            index_http_status INTEGER,
            index_retrieved_at TEXT,
            index_raw_path TEXT,
            index_byte_sha256 TEXT,
            attempts INTEGER NOT NULL DEFAULT 0,
            error TEXT
        );
        CREATE INDEX IF NOT EXISTS sec_filing_queue_status_idx
            ON sec_filing_queue(status, filing_date);

        CREATE TABLE IF NOT EXISTS sec_filing_tickers (
            accession TEXT NOT NULL REFERENCES sec_filing_queue(accession),
            ticker TEXT NOT NULL,
            company_name TEXT NOT NULL,
            sector TEXT,
            PRIMARY KEY(accession, ticker)
        );

        CREATE TABLE IF NOT EXISTS sec_documents (
            document_id INTEGER PRIMARY KEY,
            accession TEXT NOT NULL REFERENCES sec_filing_queue(accession),
            cik INTEGER NOT NULL,
            form TEXT NOT NULL,
            accepted_at TEXT NOT NULL,
            document_name TEXT NOT NULL,
            sequence INTEGER,
            description TEXT,
            exhibit_type TEXT,
            is_primary INTEGER NOT NULL,
            selection_reason TEXT NOT NULL,
            source_url TEXT NOT NULL UNIQUE,
            declared_size_bytes INTEGER,
            status TEXT NOT NULL DEFAULT 'pending'
                CHECK(status IN ('pending','ok','empty','http_error','error')),
            attempts INTEGER NOT NULL DEFAULT 0,
            http_status INTEGER,
            retrieved_at TEXT,
            raw_path TEXT,
            byte_sha256 TEXT,
            byte_count INTEGER,
            raw_text_sha256 TEXT,
            article_id INTEGER REFERENCES articles(id),
            error TEXT,
            UNIQUE(accession, document_name)
        );
        CREATE INDEX IF NOT EXISTS sec_documents_status_idx
            ON sec_documents(status, accepted_at);
        CREATE INDEX IF NOT EXISTS sec_documents_accession_idx
            ON sec_documents(accession);

        CREATE VIEW IF NOT EXISTS sec_chunk_eligibility AS
        SELECT d.document_id, d.accession, d.cik, d.form, d.accepted_at,
               d.exhibit_type, d.selection_reason, d.source_url, d.article_id,
               a.quality_status, a.canonical_article_id, a.cleaning_version
        FROM sec_documents d
        JOIN articles a ON a.id=d.article_id
        WHERE d.status='ok' AND a.quality_status='usable';
        """
    )
    db.commit()


def connect_archive(path: Path) -> sqlite3.Connection:
    db = connect_articles(path)
    db.execute("PRAGMA busy_timeout=60000")
    initialize(db)
    return db


def plan_filings(
    archive: sqlite3.Connection,
    source: sqlite3.Connection,
    since: str,
    until: str | None,
    forms: Iterable[str],
    tickers: Iterable[str],
    limit: int | None,
    newest_first: bool,
) -> int:
    forms = tuple(forms)
    tickers = tuple(value.upper() for value in tickers)
    placeholders = ",".join("?" for _ in forms)
    clauses = [f"f.form IN ({placeholders})", "f.filing_date>=?"]
    params: list[object] = [*forms, since]
    if until:
        clauses.append("f.filing_date<=?")
        params.append(until)
    if tickers:
        ticker_placeholders = ",".join("?" for _ in tickers)
        clauses.append(
            f"EXISTS (SELECT 1 FROM entities e WHERE e.cik=f.cik "
            f"AND e.ticker IN ({ticker_placeholders}))"
        )
        params.extend(tickers)
    order = "DESC" if newest_first else "ASC"
    sql = f"""
        SELECT f.accession,f.cik,f.filing_date,f.accepted_at,
               f.available_at_quality,f.form,f.primary_document,f.retrieved_at
        FROM sec_filings f
        WHERE {' AND '.join(clauses)}
        ORDER BY f.filing_date {order},f.accepted_at {order},f.accession {order}
    """
    rows = source.execute(sql, params)
    existing = {
        row[0] for row in archive.execute("SELECT accession FROM sec_filing_queue")
    }
    added = 0
    for row in rows:
        accession, cik, filing_date, accepted_at, quality, form, primary, retrieved = row
        if accession in existing:
            continue
        if limit is not None and added >= limit:
            break
        cursor = archive.execute(
            """
            INSERT OR IGNORE INTO sec_filing_queue(
                accession,cik,filing_date,accepted_at,available_at_quality,
                form,primary_document,source_metadata_retrieved_at,index_url
            ) VALUES(?,?,?,?,?,?,?,?,?)
            """,
            (
                accession,
                cik,
                filing_date,
                accepted_at,
                quality,
                form,
                primary,
                retrieved,
                filing_index_url(cik, accession),
            ),
        )
        added += int(cursor.rowcount > 0)
        existing.add(accession)
        entity_rows = source.execute(
            "SELECT ticker,company_name,sector FROM entities WHERE cik=? ORDER BY ticker",
            (cik,),
        ).fetchall()
        archive.executemany(
            """
            INSERT OR REPLACE INTO sec_filing_tickers(
                accession,ticker,company_name,sector
            ) VALUES(?,?,?,?)
            """,
            ((accession, *entity) for entity in entity_rows),
        )
    archive.commit()
    return added


def safe_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)


def write_raw(raw_dir: Path, relative: Path, content: bytes) -> Path:
    destination = raw_dir / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent, prefix=".tmp-", suffix=".gz", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        with gzip.open(temporary, "wb") as output:
            output.write(content)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def discover_filings(
    db: sqlite3.Connection,
    client: SecClient,
    raw_dir: Path,
    limit: int | None,
    workers: int = 1,
    download_workers: int | None = None,
) -> tuple[int, int]:
    sql = """
        SELECT accession,cik,primary_document,index_url
        FROM sec_filing_queue WHERE status='pending'
        ORDER BY filing_date DESC,accepted_at DESC,accession DESC
    """
    params: tuple[object, ...] = ()
    if limit is not None:
        sql += " LIMIT ?"
        params = (limit,)
    rows = db.execute(sql, params).fetchall()
    def download(row: tuple[object, ...]) -> tuple[object, ...]:
        accession, cik, primary, url = row
        content, http_status, error = client.get(url)
        retrieved_at = utc_now()
        return (
            accession, cik, primary, url, content, http_status, error,
            retrieved_at,
        )

    def process(downloaded: tuple[object, ...]) -> tuple[object, ...]:
        (
            accession, cik, primary, url, content, http_status, error,
            retrieved_at,
        ) = downloaded
        digest = sha256_bytes(content) if content else None
        raw_path = None
        if content:
            raw_path = write_raw(
                raw_dir,
                Path(str(cik)) / accession_compact(accession) / "_index.html.gz",
                content,
            )
        choices = choose_documents(content, primary) if not error else ()
        return (
            accession, cik, url, http_status, error, retrieved_at, digest,
            raw_path, choices,
        )

    network_workers = download_workers or workers
    iterator = staged_completion_order_map(
        download, process, rows, network_workers, workers,
        getattr(client, "cancel", None),
    )

    started = time.monotonic()
    limiter = getattr(client, "limiter", None)
    initial_requests = limiter.count() if limiter is not None else None
    discovered = failed = 0
    for number, result in enumerate(iterator, 1):
            (
                accession, cik, url, http_status, error, retrieved_at, digest,
                raw_path, choices,
            ) = result
            if error:
                status = "http_error" if http_status else "error"
                db.execute(
                    """
                    UPDATE sec_filing_queue SET status=?,index_http_status=?,
                        index_retrieved_at=?,index_raw_path=?,index_byte_sha256=?,
                        attempts=attempts+1,error=? WHERE accession=?
                    """,
                    (status, http_status, retrieved_at,
                     str(raw_path) if raw_path else None, digest, error, accession),
                )
                failed += 1
            else:
                for choice in choices:
                    source_url = document_url(cik, accession, choice.document_name)
                    db.execute(
                        """
                        INSERT INTO sec_documents(
                            accession,cik,form,accepted_at,document_name,sequence,
                            description,exhibit_type,is_primary,selection_reason,
                            source_url,declared_size_bytes
                        )
                        SELECT accession,cik,form,accepted_at,?,?,?,?,?,?,?,?
                        FROM sec_filing_queue WHERE accession=?
                        ON CONFLICT(accession,document_name) DO UPDATE SET
                            description=excluded.description,
                            exhibit_type=excluded.exhibit_type,
                            is_primary=excluded.is_primary,
                            selection_reason=excluded.selection_reason,
                            declared_size_bytes=excluded.declared_size_bytes
                        """,
                        (
                            choice.document_name,
                            choice.sequence,
                            choice.description,
                            choice.exhibit_type,
                            int(choice.is_primary),
                            choice.selection_reason,
                            source_url,
                            choice.size_bytes,
                            accession,
                        ),
                    )
                status = "discovered" if choices else "no_documents"
                db.execute(
                    """
                    UPDATE sec_filing_queue SET status=?,index_http_status=?,
                        index_retrieved_at=?,index_raw_path=?,index_byte_sha256=?,
                        attempts=attempts+1,error=NULL WHERE accession=?
                    """,
                    (status, http_status, retrieved_at, str(raw_path), digest, accession),
                )
                discovered += 1
            db.commit()
            print(
                f"Discovered {number:,}/{len(rows):,} filings | "
                f"ok {discovered:,} | failed {failed:,} | "
                f"{rate_and_eta(number, len(rows), started, 'filings')}"
                f"{request_rate(client, initial_requests, started)}",
                flush=True,
            )
    return discovered, failed


def fallback_sec_text(content: bytes) -> tuple[str | None, str | None]:
    """Extract SEC prose/tables once without multiplying nested article roots."""
    soup = BeautifulSoup(content, "html.parser")
    title = soup.title.get_text(" ", strip=True) if soup.title else None
    for tag in soup.select("script,style,noscript,svg"):
        tag.decompose()
    blocks: list[str] = []
    seen: set[str] = set()
    current_chars = 0

    def append_block(value: str) -> bool:
        nonlocal current_chars
        text = " ".join(value.split())
        normalized = text.casefold()
        if len(text) < 2 or normalized in seen:
            return False
        seen.add(normalized)
        remaining = MAX_SEC_TEXT_CHARS - current_chars
        if remaining <= 0:
            return True
        if len(text) > remaining:
            blocks.append(text[:remaining])
            current_chars = MAX_SEC_TEXT_CHARS
            return True
        blocks.append(text)
        current_chars += len(text) + 2
        return False

    truncated = False
    # Each paragraph/list item is visited once.  For tables, retain only leaf
    # rows so an outer SEC layout table cannot recursively repeat all contents.
    for tag in soup.find_all(["p", "li"]):
        text = " ".join(tag.get_text(" ", strip=True).split())
        if len(text) >= 25 and append_block(text):
            truncated = True
            break
    if not truncated:
        for row in soup.find_all("tr"):
            if row.find("tr") is not None:
                continue
            text = " ".join(row.get_text(" ", strip=True).split())
            if len(text) >= 10 and append_block(text):
                truncated = True
                break
    if not blocks:
        for line in soup.get_text("\n").splitlines():
            if len(line.strip()) >= 2 and append_block(line):
                truncated = True
                break
    text = "\n\n".join(blocks)
    if truncated:
        text = text[:MAX_SEC_TEXT_CHARS] + TRUNCATION_NOTICE
    return text or None, title


def extract_sec_text(content: bytes, description: str | None) -> tuple[str | None, str | None]:
    if len(content) >= LARGE_SEC_HTML_BYTES:
        text, title = fallback_sec_text(content)
        return text, title or (description.strip() if description else None)

    extracted = extract_article(content)
    text = extracted["text"]
    title = extracted["title"]
    # Nested article/body selectors can multiply SEC table contents.  Replace
    # any suspicious expansion with the single-pass SEC extractor.
    suspicious_expansion = bool(
        text and (
            len(text) > MAX_SEC_TEXT_CHARS
            or len(text) > max(200_000, len(content) * 20)
        )
    )
    if not text or len(text) < 300 or suspicious_expansion:
        fallback_text, fallback_title = fallback_sec_text(content)
        if suspicious_expansion or len(fallback_text or "") > len(text or ""):
            text = fallback_text
        title = title or fallback_title
    if text and len(text) > MAX_SEC_TEXT_CHARS + len(TRUNCATION_NOTICE):
        text = text[:MAX_SEC_TEXT_CHARS] + TRUNCATION_NOTICE
    title = title or (description.strip() if description else None)
    return text, title


def process_document_download(
    downloaded: tuple[object, ...],
    raw_dir: Path,
) -> tuple[object, ...]:
    """Compress and extract one downloaded filing outside the parent process."""
    (
        document_id, accession, cik, form, accepted_at, name,
        description, exhibit_type, selection_reason, url, content,
        http_status, error, retrieved_at,
    ) = downloaded
    raw_path = text = title = byte_digest = text_digest = None
    if not error:
        raw_path = write_raw(
            raw_dir,
            Path(str(cik)) / accession_compact(str(accession))
            / f"{safe_component(str(name))}.gz",
            content,
        )
        text, title = extract_sec_text(content, description)
        byte_digest = sha256_bytes(content)
        text_digest = sha256_text(text) if text else None
    return (
        document_id, accession, cik, form, accepted_at, name, description,
        exhibit_type, selection_reason, url, content, http_status, error,
        retrieved_at, raw_path, text, title, byte_digest, text_digest,
    )


def fetch_documents(
    db: sqlite3.Connection,
    client: SecClient,
    raw_dir: Path,
    limit: int | None,
    workers: int = 1,
    download_workers: int | None = None,
) -> tuple[int, int]:
    sql = """
        SELECT document_id,accession,cik,form,accepted_at,document_name,
               description,exhibit_type,selection_reason,source_url
        FROM sec_documents WHERE status='pending'
        ORDER BY accepted_at DESC,document_id
    """
    params: tuple[object, ...] = ()
    if limit is not None:
        sql += " LIMIT ?"
        params = (limit,)
    rows = db.execute(sql, params).fetchall()
    def download(row: tuple[object, ...]) -> tuple[object, ...]:
        (
            document_id, accession, cik, form, accepted_at, name,
            description, exhibit_type, selection_reason, url,
        ) = row
        content, http_status, error = client.get(url)
        retrieved_at = utc_now()
        return (*row, content, http_status, error, retrieved_at)

    process = partial(process_document_download, raw_dir=raw_dir)

    network_workers = download_workers or workers
    iterator = staged_completion_order_map(
        download, process, rows, network_workers, workers,
        getattr(client, "cancel", None),
        process_pool=True,
    )

    started = time.monotonic()
    limiter = getattr(client, "limiter", None)
    initial_requests = limiter.count() if limiter is not None else None
    fetched = failed = 0
    for number, result in enumerate(iterator, 1):
            (
                document_id, accession, cik, form, accepted_at, name, description,
                exhibit_type, selection_reason, url, content, http_status, error,
                retrieved_at, raw_path, text, title, byte_digest, text_digest,
            ) = result
            if error:
                status = "http_error" if http_status else "error"
                db.execute(
                    """
                    UPDATE sec_documents SET status=?,attempts=attempts+1,
                        http_status=?,retrieved_at=?,error=? WHERE document_id=?
                    """,
                    (status, http_status, retrieved_at, error, document_id),
                )
                failed += 1
            else:
                status = "ok" if text else "empty"
                try:
                    db.execute(
                        """
                        INSERT INTO articles(
                            source_url,final_url,domain,title,title_raw,article_text,
                            article_text_raw,status,http_status,fetched_at,published_at,
                            language
                        ) VALUES(?,?,?,?,?,?,?,?,?,?,?,'en')
                        ON CONFLICT(source_url) DO UPDATE SET
                            final_url=excluded.final_url,
                            title=excluded.title,
                            title_raw=excluded.title_raw,
                            article_text=excluded.article_text,
                            article_text_raw=excluded.article_text_raw,
                            status=excluded.status,
                            http_status=excluded.http_status,
                            fetched_at=excluded.fetched_at,
                            published_at=excluded.published_at,
                            article_text_clean=NULL,
                            cleaning_version=NULL,
                            quality_status=NULL
                        """,
                        (
                            url, url, "sec.gov", title, title, text, text, status,
                            http_status, retrieved_at, accepted_at,
                        ),
                    )
                except sqlite3.DataError as exc:
                    # The complete compressed response remains available even
                    # if a future pathological extraction exceeds SQLite's
                    # configured value limit.  Mark it terminal so one document
                    # cannot crash every resumed batch.
                    error_message = f"extracted text rejected by SQLite: {exc}"
                    db.execute(
                        """
                        UPDATE sec_documents SET status='rejected_oversize',
                            attempts=attempts+1,http_status=?,retrieved_at=?,
                            raw_path=?,byte_sha256=?,byte_count=?,
                            raw_text_sha256=?,error=? WHERE document_id=?
                        """,
                        (
                            http_status, retrieved_at, str(raw_path), byte_digest,
                            len(content), text_digest, error_message, document_id,
                        ),
                    )
                    db.execute(
                        """
                        UPDATE sec_filing_queue
                        SET status=CASE
                            WHEN EXISTS(
                                SELECT 1 FROM sec_documents d
                                WHERE d.accession=sec_filing_queue.accession
                                  AND d.status='pending'
                            ) THEN 'discovered'
                            ELSE 'complete'
                        END
                        WHERE accession=?
                        """,
                        (accession,),
                    )
                    db.commit()
                    failed += 1
                    print(
                        f"Rejected oversized SEC document {document_id} "
                        f"({url}); raw source retained",
                        flush=True,
                    )
                    continue
                article_id = db.execute(
                    "SELECT id FROM articles WHERE source_url=?", (url,)
                ).fetchone()[0]
                ticker_row = db.execute(
                    """
                    SELECT ticker FROM sec_filing_tickers
                    WHERE accession=? ORDER BY ticker LIMIT 1
                    """,
                    (accession,),
                ).fetchone()
                event_category = (
                    "sec_press_release"
                    if selection_reason in {"exhibit_99", "press_release_description"}
                    else f"sec_{form.lower().replace('/', '_').replace('-', '')}"
                )
                if db.execute(
                    "SELECT 1 FROM events WHERE article_id=? LIMIT 1", (article_id,)
                ).fetchone() is None:
                    db.execute(
                        """
                        INSERT INTO events(date,event_category,primary_actor,article_id)
                        VALUES(?,?,?,?)
                        """,
                        (accepted_at[:10], event_category,
                         ticker_row[0] if ticker_row else str(cik), article_id),
                    )
                db.execute(
                    """
                    UPDATE sec_documents SET status=?,attempts=attempts+1,
                        http_status=?,retrieved_at=?,raw_path=?,byte_sha256=?,
                        byte_count=?,raw_text_sha256=?,article_id=?,error=NULL
                    WHERE document_id=?
                    """,
                    (
                        status, http_status, retrieved_at, str(raw_path), byte_digest,
                        len(content), text_digest, article_id, document_id,
                    ),
                )
                fetched += 1
            db.execute(
                """
                UPDATE sec_filing_queue
                SET status=CASE
                    WHEN EXISTS(
                        SELECT 1 FROM sec_documents d
                        WHERE d.accession=sec_filing_queue.accession
                          AND d.status='pending'
                    ) THEN 'discovered'
                    ELSE 'complete'
                END
                WHERE accession=?
                """,
                (accession,),
            )
            db.commit()
            print(
                f"Fetched {number:,}/{len(rows):,} SEC documents | "
                f"ok {fetched:,} | failed {failed:,} | "
                f"{rate_and_eta(number, len(rows), started, 'documents')}"
                f"{request_rate(client, initial_requests, started)}",
                flush=True,
            )
    return fetched, failed


def retry_failed(db: sqlite3.Connection) -> None:
    db.execute(
        """
        UPDATE sec_filing_queue SET status='pending',error=NULL
        WHERE status IN ('http_error','error')
        """
    )
    db.execute(
        """
        UPDATE sec_documents SET status='pending',error=NULL
        WHERE status IN ('http_error','error')
        """
    )
    db.commit()


def reextract_oversized_article(
    row: tuple[object, ...],
) -> tuple[object, ...]:
    document_id, article_id, raw_path, description = row
    try:
        with gzip.open(Path(str(raw_path)), "rb") as source:
            content = source.read()
        text, extracted_title = extract_sec_text(content, description)
        if not text:
            raise ValueError("safe extraction produced no text")
        return (
            document_id, article_id, text, extracted_title,
            sha256_text(text), None,
        )
    except (OSError, ValueError) as exc:
        return document_id, article_id, None, None, None, str(exc)


def repair_oversized_articles(
    db: sqlite3.Connection,
    workers: int = 4,
) -> tuple[int, int]:
    """Re-extract stored outliers from retained raw HTML using the safe parser."""
    rows = db.execute(
        """
        SELECT d.document_id,d.article_id,d.raw_path,d.description
        FROM sec_documents d
        JOIN articles a ON a.id=d.article_id
        WHERE d.raw_path IS NOT NULL
          AND length(a.article_text_raw)>?
        ORDER BY d.document_id
        """,
        (MAX_SEC_TEXT_CHARS + len(TRUNCATION_NOTICE),),
    ).fetchall()
    if not rows:
        return 0, 0

    print(
        f"Repairing {len(rows):,} oversized SEC texts from retained raw filings "
        f"(cap {MAX_SEC_TEXT_CHARS:,} characters)...",
        flush=True,
    )
    repaired = failed = 0
    iterator = staged_completion_order_map(
        lambda row: row,
        reextract_oversized_article,
        rows,
        download_workers=1,
        process_workers=workers,
        process_pool=True,
    )
    for number, result in enumerate(iterator, 1):
        (
            document_id, article_id, text, extracted_title, digest, error,
        ) = result
        if error is None:
            db.execute(
                """
                UPDATE articles SET
                    title=COALESCE(?,title),
                    title_raw=COALESCE(?,title_raw),
                    article_text=?,
                    article_text_raw=?,
                    article_text_clean=NULL,
                    cleaning_version=NULL,
                    quality_status=NULL
                WHERE id=?
                """,
                (
                    extracted_title, extracted_title, text, text, article_id,
                ),
            )
            db.execute(
                """
                UPDATE sec_documents SET raw_text_sha256=?,error=NULL
                WHERE document_id=?
                """,
                (digest, document_id),
            )
            repaired += 1
        else:
            failed += 1
            db.execute(
                "UPDATE sec_documents SET error=? WHERE document_id=?",
                (f"oversized text repair failed: {error}", document_id),
            )
        if number % 25 == 0:
            db.commit()
            print(
                f"Repaired {number:,}/{len(rows):,} oversized texts | "
                f"ok {repaired:,} | failed {failed:,}",
                flush=True,
            )
    db.commit()
    return repaired, failed


def process_archive(db: sqlite3.Connection, workers: int = 4) -> None:
    # Repeated content on sec.gov is often a legitimate exhibit filed by more
    # than one issuer/document, not a publisher's dead-URL fallback page.
    repaired, repair_failed = repair_oversized_articles(db, workers)
    if repaired or repair_failed:
        print(
            f"Oversized SEC text repair complete | "
            f"ok {repaired:,} | failed {repair_failed:,}",
            flush=True,
        )
    clean_corpus(db, reject_repeated_domain_fallback=False)
    stats = update_effective_dates(db)
    print(
        "SEC cleaning complete | "
        f"{stats.get('articles_updated', 0):,} dated articles | "
        f"{stats.get('canonical_dates', 0):,} canonical documents",
        flush=True,
    )


def print_summary(db: sqlite3.Connection, db_path: Path, raw_dir: Path) -> None:
    filings = db.execute("SELECT COUNT(*) FROM sec_filing_queue").fetchone()[0]
    docs = db.execute("SELECT COUNT(*) FROM sec_documents").fetchone()[0]
    articles = db.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
    raw_bytes = db.execute(
        "SELECT COALESCE(SUM(byte_count),0) FROM sec_documents WHERE status='ok'"
    ).fetchone()[0]
    stored_bytes = sum(path.stat().st_size for path in raw_dir.rglob("*.gz")) if raw_dir.exists() else 0
    print("\nSEC text archive summary")
    print(f"  Database: {db_path} ({db_path.stat().st_size / 1024**2:.2f} MiB)")
    print(f"  Planned filings: {filings:,}")
    for status, count in db.execute(
        "SELECT status,COUNT(*) FROM sec_filing_queue GROUP BY status ORDER BY status"
    ):
        print(f"  filing {status}: {count:,}")
    print(f"  Selected documents: {docs:,}")
    for status, count in db.execute(
        "SELECT status,COUNT(*) FROM sec_documents GROUP BY status ORDER BY status"
    ):
        print(f"  document {status}: {count:,}")
    for reason, count in db.execute(
        "SELECT selection_reason,COUNT(*) FROM sec_documents GROUP BY selection_reason ORDER BY selection_reason"
    ):
        print(f"  selected {reason}: {count:,}")
    for status, count in db.execute(
        """
        SELECT quality_status,COUNT(*) FROM articles
        WHERE quality_status IS NOT NULL GROUP BY quality_status ORDER BY quality_status
        """
    ):
        print(f"  quality {status}: {count:,}")
    print(f"  Article rows: {articles:,}")
    print(f"  Raw source bytes: {raw_bytes / 1024**2:.2f} MiB")
    print(f"  Compressed raw storage: {stored_bytes / 1024**2:.2f} MiB")


def parse_csv_values(values: str) -> tuple[str, ...]:
    return tuple(value.strip() for value in values.split(",") if value.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("plan", "sync", "process", "summary"),
        help="plan metadata, fetch bounded work, clean text, or report state",
    )
    parser.add_argument("--source-db", type=Path, default=DEFAULT_SOURCE_DB)
    parser.add_argument("--db", type=Path, default=DEFAULT_ARCHIVE_DB)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--since", default="2015-01-01")
    parser.add_argument("--until")
    parser.add_argument("--forms", default=",".join(FORMS))
    parser.add_argument("--tickers", default="")
    parser.add_argument(
        "--skip-plan", action="store_true",
        help="resume already queued work without adding another planning batch",
    )
    parser.add_argument("--limit-filings", type=int)
    parser.add_argument("--max-documents", type=int)
    parser.add_argument("--oldest-first", action="store_true")
    parser.add_argument("--requests-per-second", type=float, default=4.0)
    parser.add_argument(
        "--workers", type=int, default=4,
        help="parallel gzip/HTML processing processes",
    )
    parser.add_argument(
        "--download-workers", type=int, default=32,
        help=(
            "parallel SEC network workers feeding the processing queue; request "
            "starts still share --requests-per-second"
        ),
    )
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()

    if args.limit_filings is not None and args.limit_filings <= 0:
        raise ValueError("--limit-filings must be positive")
    if args.max_documents is not None and args.max_documents <= 0:
        raise ValueError("--max-documents must be positive")
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    if args.download_workers <= 0:
        raise ValueError("--download-workers must be positive")
    forms = parse_csv_values(args.forms)
    if not forms or any(form not in FORMS for form in forms):
        raise ValueError(f"--forms must be drawn from {FORMS}")

    with connect_archive(args.db) as archive:
        if args.retry_failed:
            retry_failed(archive)
        if args.command in {"plan", "sync"} and not args.skip_plan:
            with sqlite3.connect(args.source_db) as source:
                added = plan_filings(
                    archive, source, args.since, args.until, forms,
                    parse_csv_values(args.tickers), args.limit_filings,
                    not args.oldest_first,
                )
            print(f"Planned {added:,} new filings", flush=True)
        if args.command == "sync":
            user_agent = default_sec_user_agent()
            if not user_agent:
                raise RuntimeError(
                    "Set SEC_USER_AGENT='Name email@example.com' before fetching SEC data"
                )
            client = SecClient(
                user_agent, args.requests_per_second, args.retries
            )
            discovered, discovery_failed = discover_filings(
                archive, client, args.raw_dir, args.limit_filings, args.workers,
                args.download_workers,
            )
            fetched, fetch_failed = fetch_documents(
                archive, client, args.raw_dir, args.max_documents, args.workers,
                args.download_workers,
            )
            print(
                f"Sync complete | filings {discovered:,} ok/{discovery_failed:,} failed | "
                f"documents {fetched:,} ok/{fetch_failed:,} failed",
                flush=True,
            )
        elif args.command == "process":
            process_archive(archive, args.workers)
        print_summary(archive, args.db, args.raw_dir)


if __name__ == "__main__":
    main()
