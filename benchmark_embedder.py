#!/usr/bin/env python3
"""Benchmark local embedding speed on a representative article sample.

This is deliberately read-only: it does not create embedding tables or retain
the generated vectors.  Use it to choose a batch size before a full run.
"""

from __future__ import annotations

import argparse
import json
import platform
import sqlite3
import time
from pathlib import Path

DEFAULT_DB = Path("historical_news.sqlite3")
DEFAULT_MODEL = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_CHUNKING_VERSION = "paragraph-v1-500-600-75"


def evenly_spaced(values: list[int], count: int) -> list[int]:
    """Return a deterministic sample spanning the entire ordered population."""
    if count <= 0:
        raise ValueError("article count must be positive")
    if len(values) <= count:
        return values
    return [values[index * len(values) // count] for index in range(count)]


def load_sample(
    db: sqlite3.Connection,
    chunking_version: str,
    article_count: int,
) -> tuple[list[tuple[int, str, int]], int, int]:
    """Load every chunk for an evenly spaced sample of canonical articles."""
    eligible = [
        row[0]
        for row in db.execute(
            """SELECT DISTINCT a.id
               FROM articles a
               JOIN article_chunks c ON c.article_id = a.id
               WHERE c.chunking_version = ?
                 AND a.quality_status = 'usable'
                 AND COALESCE(a.canonical_article_id, a.id) = a.id
               ORDER BY a.id""",
            (chunking_version,),
        )
    ]
    selected = evenly_spaced(eligible, article_count)
    if not selected:
        raise RuntimeError(
            f"no canonical chunks found for chunking version {chunking_version!r}"
        )
    placeholders = ",".join("?" for _ in selected)
    rows = db.execute(
        f"""SELECT article_id, embedding_text, token_count
            FROM article_chunks
            WHERE chunking_version = ? AND article_id IN ({placeholders})
            ORDER BY article_id, chunk_index""",
        (chunking_version, *selected),
    ).fetchall()
    return rows, len(eligible), len(selected)


def choose_device(requested: str) -> str:
    import torch

    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def memory_stats(device: str) -> dict[str, float]:
    import os

    result: dict[str, float] = {}
    try:
        import psutil

        result["process_rss_gib"] = psutil.Process(os.getpid()).memory_info().rss / 2**30
    except ImportError:
        pass
    if device == "mps":
        import torch

        result["mps_allocated_gib"] = torch.mps.current_allocated_memory() / 2**30
        result["mps_driver_gib"] = torch.mps.driver_allocated_memory() / 2**30
    return result


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    if seconds < 3600:
        return f"{seconds / 60:.1f} minutes"
    return f"{seconds / 3600:.1f} hours"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--chunking-version", default=DEFAULT_CHUNKING_VERSION)
    parser.add_argument("--articles", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    parser.add_argument(
        "--local-files-only", action="store_true",
        help="fail instead of downloading model weights that are not cached",
    )
    parser.add_argument("--json", type=Path, help="optionally save the report as JSON")
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")

    uri = f"file:{args.db.resolve()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as db:
        rows, corpus_articles, sampled_articles = load_sample(
            db, args.chunking_version, args.articles
        )
        corpus_chunks = db.execute(
            """SELECT COUNT(*) FROM article_chunks c JOIN articles a ON a.id=c.article_id
               WHERE c.chunking_version=? AND a.quality_status='usable'
                 AND COALESCE(a.canonical_article_id,a.id)=a.id""",
            (args.chunking_version,),
        ).fetchone()[0]

    texts = [row[1] for row in rows]
    input_tokens = sum(row[2] for row in rows)
    print(
        f"Loaded {len(texts):,} chunks ({input_tokens:,} tokens) from "
        f"{sampled_articles:,} canonical articles.",
        flush=True,
    )

    import torch
    from sentence_transformers import SentenceTransformer

    device = choose_device(args.device)
    model_kwargs = {"local_files_only": args.local_files_only}
    if device in ("mps", "cuda"):
        model_kwargs["torch_dtype"] = torch.float16
    print(f"Loading {args.model} on {device}...", flush=True)
    load_started = time.perf_counter()
    model = SentenceTransformer(args.model, device=device, model_kwargs=model_kwargs)
    model.max_seq_length = 600
    load_seconds = time.perf_counter() - load_started

    warmup = texts[: min(args.batch_size, len(texts))]
    print(f"Warming up with {len(warmup)} chunks...", flush=True)
    model.encode(
        warmup, batch_size=args.batch_size, normalize_embeddings=True,
        convert_to_numpy=True, show_progress_bar=False,
    )
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()

    print(f"Benchmarking batch size {args.batch_size}...", flush=True)
    run_started = time.perf_counter()
    vectors = model.encode(
        texts, batch_size=args.batch_size, normalize_embeddings=True,
        convert_to_numpy=True, show_progress_bar=True,
    )
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()
    run_seconds = time.perf_counter() - run_started

    chunk_rate = len(texts) / run_seconds
    article_rate = sampled_articles / run_seconds
    token_rate = input_tokens / run_seconds
    dimension = int(vectors.shape[1])
    projected_seconds = corpus_chunks / chunk_rate
    vector_bytes_f16 = corpus_chunks * dimension * 2
    revision = getattr(model[0].auto_model.config, "_commit_hash", None)
    report = {
        "model": args.model,
        "model_revision": revision,
        "device": device,
        "dtype": "float16" if device in ("mps", "cuda") else "float32",
        "host": platform.platform(),
        "batch_size": args.batch_size,
        "load_seconds": load_seconds,
        "sample_articles": sampled_articles,
        "sample_chunks": len(texts),
        "sample_tokens": input_tokens,
        "run_seconds": run_seconds,
        "chunks_per_second": chunk_rate,
        "articles_per_second": article_rate,
        "tokens_per_second": token_rate,
        "embedding_dimension": dimension,
        "canonical_corpus_articles": corpus_articles,
        "canonical_corpus_chunks": corpus_chunks,
        "projected_corpus_seconds": projected_seconds,
        "projected_float16_vector_gib": vector_bytes_f16 / 2**30,
        "memory": memory_stats(device),
    }

    print("\nBenchmark result")
    print(f"  Model load: {format_duration(load_seconds)}")
    print(f"  Embedding run: {format_duration(run_seconds)}")
    print(f"  Throughput: {chunk_rate:.2f} chunks/s | {article_rate:.2f} articles/s | {token_rate:,.0f} tokens/s")
    print(f"  Vector shape: {len(texts):,} x {dimension:,}")
    print(f"  Full canonical corpus: {corpus_articles:,} articles / {corpus_chunks:,} chunks")
    print(f"  Projected embedding time: {format_duration(projected_seconds)}")
    print(f"  Projected float16 vectors: {vector_bytes_f16 / 2**30:.2f} GiB (before indexes/metadata)")
    for name, value in report["memory"].items():
        print(f"  {name}: {value:.2f} GiB")
    if args.json:
        args.json.write_text(json.dumps(report, indent=2) + "\n")
        print(f"  Report saved to {args.json}")


if __name__ == "__main__":
    main()
