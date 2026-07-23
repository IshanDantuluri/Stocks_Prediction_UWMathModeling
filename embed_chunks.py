#!/usr/bin/env python3
"""Embed canonical article chunks into a resumable SQLite sidecar database."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import time
from collections import deque
from pathlib import Path

import numpy as np

DEFAULT_SOURCE = Path("historical_news.sqlite3")
DEFAULT_OUTPUT = Path("news_embeddings.sqlite3")
DEFAULT_MODEL = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_CHUNKING_VERSION = "paragraph-v1-500-600-75"


def initialize_output(db: sqlite3.Connection) -> None:
    db.executescript(
        """
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=NORMAL;
        CREATE TABLE IF NOT EXISTS embedding_configs (
            id INTEGER PRIMARY KEY,
            config_hash TEXT NOT NULL UNIQUE,
            model_name TEXT NOT NULL,
            model_revision TEXT NOT NULL,
            dimension INTEGER NOT NULL,
            storage_dtype TEXT NOT NULL,
            normalized INTEGER NOT NULL,
            chunking_version TEXT NOT NULL,
            config_json TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS chunk_embeddings (
            config_id INTEGER NOT NULL REFERENCES embedding_configs(id),
            chunk_id INTEGER NOT NULL,
            article_id INTEGER NOT NULL,
            chunk_hash TEXT NOT NULL,
            vector BLOB NOT NULL,
            embedded_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (config_id, chunk_hash)
        );
        CREATE INDEX IF NOT EXISTS chunk_embeddings_article_idx
            ON chunk_embeddings(config_id, article_id);
        """
    )


def get_or_create_config(
    db: sqlite3.Connection,
    model_name: str,
    model_revision: str,
    dimension: int,
    chunking_version: str,
) -> int:
    config = {
        "model_name": model_name,
        "model_revision": model_revision,
        "dimension": dimension,
        "storage_dtype": "float16-le",
        "normalized": True,
        "chunking_version": chunking_version,
    }
    config_json = json.dumps(config, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(config_json.encode()).hexdigest()
    db.execute(
        """INSERT OR IGNORE INTO embedding_configs
           (config_hash,model_name,model_revision,dimension,storage_dtype,
            normalized,chunking_version,config_json)
           VALUES (?,?,?,?,?,?,?,?)""",
        (digest, model_name, model_revision, dimension, "float16-le", 1,
         chunking_version, config_json),
    )
    row = db.execute(
        "SELECT id FROM embedding_configs WHERE config_hash=?", (digest,)
    ).fetchone()
    assert row is not None
    return row[0]


def canonical_chunk_query(limit: int | None = None) -> tuple[str, tuple]:
    sql = """SELECT c.id,c.article_id,c.chunk_hash,c.embedding_text,c.token_count
             FROM article_chunks c
             JOIN articles a ON a.id=c.article_id
             WHERE c.chunking_version=?
               AND a.quality_status='usable'
               AND COALESCE(a.canonical_article_id,a.id)=a.id
             ORDER BY c.id"""
    if limit is None:
        return sql, ()
    return sql + " LIMIT ?", (limit,)


def encode_with_backoff(model, rows: list[tuple], batch_size: int, device: str):
    """Encode a prefix, reducing the batch after a CUDA out-of-memory error."""
    import torch

    size = min(batch_size, len(rows))
    while True:
        try:
            vectors = model.encode(
                [row[3] for row in rows[:size]],
                batch_size=size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            return rows[:size], vectors, size
        except RuntimeError as error:
            if "out of memory" not in str(error).lower() or size == 1:
                raise
            size = max(1, size // 2)
            if device == "cuda":
                torch.cuda.empty_cache()
            print(f"GPU out of memory; retrying with batch size {size}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--chunking-version", default=DEFAULT_CHUNKING_VERSION)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--limit", type=int, help="maximum canonical chunks to consider")
    parser.add_argument("--device", choices=("cuda", "mps", "cpu"), default="cuda")
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")

    import torch
    from sentence_transformers import SentenceTransformer

    dtype = torch.float16 if args.device in ("cuda", "mps") else torch.float32
    print(f"Loading {args.model} on {args.device}...", flush=True)
    model = SentenceTransformer(
        args.model, device=args.device, model_kwargs={"torch_dtype": dtype}
    )
    model.max_seq_length = 600
    dimension = int(model.get_sentence_embedding_dimension())
    revision = getattr(model[0].auto_model.config, "_commit_hash", None) or "unknown"
    print(f"Model revision {revision}; {dimension:,}-dimensional vectors", flush=True)

    source_uri = f"file:{args.source.resolve()}?mode=ro"
    with sqlite3.connect(source_uri, uri=True) as source, sqlite3.connect(args.output) as output:
        initialize_output(output)
        config_id = get_or_create_config(
            output, args.model, revision, dimension, args.chunking_version
        )
        output.commit()
        completed = {
            row[0] for row in output.execute(
                "SELECT chunk_hash FROM chunk_embeddings WHERE config_id=?", (config_id,)
            )
        }

        query, tail_params = canonical_chunk_query(args.limit)
        params = (args.chunking_version, *tail_params)
        total = source.execute(
            """SELECT COUNT(*) FROM article_chunks c JOIN articles a ON a.id=c.article_id
               WHERE c.chunking_version=? AND a.quality_status='usable'
                 AND COALESCE(a.canonical_article_id,a.id)=a.id""",
            (args.chunking_version,),
        ).fetchone()[0]
        if args.limit is not None:
            total = min(total, args.limit)
        print(
            f"Corpus contains {total:,} canonical chunks; "
            f"{len(completed):,} already embedded for this configuration.", flush=True,
        )

        pending: deque[tuple] = deque()
        cursor = source.execute(query, params)
        examined = written = 0
        current_batch = args.batch_size
        started = time.monotonic()
        exhausted = False
        while pending or not exhausted:
            while len(pending) < current_batch and not exhausted:
                rows = cursor.fetchmany(max(1000, current_batch))
                if not rows:
                    exhausted = True
                    break
                examined += len(rows)
                pending.extend(row for row in rows if row[2] not in completed)
            if not pending:
                continue
            candidates = list(pending)[:current_batch]
            encoded_rows, vectors, successful_size = encode_with_backoff(
                model, candidates, current_batch, args.device
            )
            current_batch = successful_size
            records = []
            for row, vector in zip(encoded_rows, vectors, strict=True):
                chunk_id, article_id, digest = row[:3]
                blob = np.asarray(vector, dtype="<f2").tobytes()
                records.append((config_id, chunk_id, article_id, digest, blob))
            output.executemany(
                """INSERT OR IGNORE INTO chunk_embeddings
                   (config_id,chunk_id,article_id,chunk_hash,vector) VALUES (?,?,?,?,?)""",
                records,
            )
            output.commit()
            for _ in encoded_rows:
                pending.popleft()
            written += len(encoded_rows)

            elapsed = max(time.monotonic() - started, 0.001)
            done = min(total, len(completed) + written)
            rate = written / elapsed
            eta = (total - done) / rate if rate else 0
            if written % max(500, current_batch) < current_batch or done == total:
                print(
                    f"Embedded {done:,}/{total:,} chunks | batch {current_batch} | "
                    f"{rate:.2f} new chunks/s | ETA {eta / 60:.1f} min",
                    flush=True,
                )

        final_count = output.execute(
            "SELECT COUNT(*) FROM chunk_embeddings WHERE config_id=?", (config_id,)
        ).fetchone()[0]
    print(f"Complete: {final_count:,} vectors stored in {args.output}")


if __name__ == "__main__":
    main()
