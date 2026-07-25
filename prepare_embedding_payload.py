#!/usr/bin/env python3
"""Create a compact, read-only-transfer copy of canonical chunk inputs."""

from __future__ import annotations

import argparse
import sqlite3
import time
from pathlib import Path


DEFAULT_SOURCE = Path("sec_text_archive.sqlite3")
DEFAULT_OUTPUT = Path("sec_embedding_input.sqlite3")
DEFAULT_VERSION = "paragraph-v1-500-600-75"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--chunking-version", default=DEFAULT_VERSION)
    parser.add_argument("--batch-size", type=int, default=10_000)
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.output.exists():
        raise FileExistsError(
            f"{args.output} already exists; remove or rename it explicitly"
        )

    source_uri = f"file:{args.source.resolve()}?mode=ro"
    with sqlite3.connect(source_uri, uri=True) as source, sqlite3.connect(
        args.output
    ) as output:
        output.executescript(
            """
            PRAGMA journal_mode=OFF;
            PRAGMA synchronous=OFF;
            PRAGMA temp_store=MEMORY;
            PRAGMA page_size=32768;
            CREATE TABLE articles(
                id INTEGER PRIMARY KEY,
                quality_status TEXT NOT NULL,
                canonical_article_id INTEGER NOT NULL
            );
            CREATE TABLE article_chunks(
                id INTEGER PRIMARY KEY,
                article_id INTEGER NOT NULL,
                chunking_version TEXT NOT NULL,
                chunk_hash TEXT NOT NULL,
                embedding_text TEXT NOT NULL,
                token_count INTEGER NOT NULL
            );
            """
        )
        total = source.execute(
            """
            SELECT COUNT(*)
            FROM article_chunks c JOIN articles a ON a.id=c.article_id
            WHERE c.chunking_version=?
              AND a.quality_status='usable'
              AND COALESCE(a.canonical_article_id,a.id)=a.id
            """,
            (args.chunking_version,),
        ).fetchone()[0]
        cursor = source.execute(
            """
            SELECT c.id,c.article_id,c.chunking_version,c.chunk_hash,
                   c.embedding_text,c.token_count
            FROM article_chunks c JOIN articles a ON a.id=c.article_id
            WHERE c.chunking_version=?
              AND a.quality_status='usable'
              AND COALESCE(a.canonical_article_id,a.id)=a.id
            ORDER BY c.id
            """,
            (args.chunking_version,),
        )
        copied = 0
        started = time.monotonic()
        while True:
            rows = cursor.fetchmany(args.batch_size)
            if not rows:
                break
            article_ids = sorted({int(row[1]) for row in rows})
            output.executemany(
                "INSERT OR IGNORE INTO articles VALUES (?,'usable',?)",
                ((article_id, article_id) for article_id in article_ids),
            )
            output.executemany(
                "INSERT INTO article_chunks VALUES (?,?,?,?,?,?)",
                rows,
            )
            output.commit()
            copied += len(rows)
            elapsed = max(time.monotonic() - started, 0.001)
            rate = copied / elapsed
            eta = (total - copied) / rate if rate else 0
            print(
                f"Copied {copied:,}/{total:,} chunks | {rate:,.0f}/s | "
                f"ETA {eta / 60:.1f} min",
                flush=True,
            )
        output.execute(
            "CREATE INDEX article_chunks_version_idx "
            "ON article_chunks(chunking_version,id)"
        )
        output.execute(
            "CREATE INDEX article_chunks_article_idx "
            "ON article_chunks(article_id)"
        )
        output.commit()
        copied_total = output.execute(
            "SELECT COUNT(*) FROM article_chunks"
        ).fetchone()[0]
        if copied_total != total:
            raise RuntimeError(
                f"payload count mismatch: copied {copied_total:,}, expected {total:,}"
            )
    print(
        f"Prepared {args.output} with {total:,} canonical chunks "
        f"({args.output.stat().st_size / 2**30:.2f} GiB)."
    )


if __name__ == "__main__":
    main()
