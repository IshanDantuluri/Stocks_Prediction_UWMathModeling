#!/usr/bin/env python3
"""Create stable, paragraph-aware chunks from cleaned news articles."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

DEFAULT_DB = Path("historical_news.sqlite3")
DEFAULT_TOKENIZER = "Qwen/Qwen3-Embedding-0.6B"
CHUNKING_VERSION = "paragraph-v1"
SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?。！？])\s+")


class Tokenizer(Protocol):
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]: ...
    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str: ...


@dataclass(frozen=True)
class Unit:
    text: str
    paragraph_index: int
    tokens: int


@dataclass(frozen=True)
class Chunk:
    index: int
    body_text: str
    embedding_text: str
    token_count: int
    paragraph_start: int
    paragraph_end: int


def token_count(tokenizer: Tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def hard_split(text: str, paragraph_index: int, limit: int, tokenizer: Tokenizer) -> list[Unit]:
    ids = tokenizer.encode(text, add_special_tokens=False)
    result = []
    for start in range(0, len(ids), limit):
        piece_ids = ids[start:start + limit]
        piece = " ".join(tokenizer.decode(piece_ids, skip_special_tokens=True).split())
        if piece:
            result.append(Unit(piece, paragraph_index, len(piece_ids)))
    return result


def split_paragraph(paragraph: str, index: int, limit: int, tokenizer: Tokenizer) -> list[Unit]:
    count = token_count(tokenizer, paragraph)
    if count <= limit:
        return [Unit(paragraph, index, count)]
    sentences = [s.strip() for s in SENTENCE_BOUNDARY.split(paragraph) if s.strip()]
    if len(sentences) <= 1:
        return hard_split(paragraph, index, limit, tokenizer)
    units = []
    current = []
    current_tokens = 0
    for sentence in sentences:
        sentence_tokens = token_count(tokenizer, sentence)
        if sentence_tokens > limit:
            if current:
                text = " ".join(current)
                units.append(Unit(text, index, current_tokens))
                current, current_tokens = [], 0
            units.extend(hard_split(sentence, index, limit, tokenizer))
        elif current and current_tokens + sentence_tokens > limit:
            text = " ".join(current)
            units.append(Unit(text, index, current_tokens))
            current, current_tokens = [sentence], sentence_tokens
        else:
            current.append(sentence)
            current_tokens += sentence_tokens
    if current:
        units.append(Unit(" ".join(current), index, current_tokens))
    return units


def make_chunks(
    title: str | None,
    article_text: str,
    tokenizer: Tokenizer,
    target_tokens: int = 500,
    max_tokens: int = 600,
    overlap_tokens: int = 75,
    minimum_tokens: int = 100,
) -> list[Chunk]:
    if not 0 <= overlap_tokens < target_tokens <= max_tokens:
        raise ValueError("require 0 <= overlap < target <= maximum")
    clean_title = " ".join((title or "Untitled article").split())
    prefix = f"Title: {clean_title}\n\n"
    prefix_tokens = token_count(tokenizer, prefix)
    body_target = target_tokens - prefix_tokens
    body_max = max_tokens - prefix_tokens
    if body_target < max(10, target_tokens // 5):
        raise ValueError("title leaves too little room for article text")

    paragraphs = [" ".join(p.split()) for p in article_text.split("\n\n") if p.strip()]
    units = []
    # Leave a small margin because tokenization is not perfectly additive when
    # separately tokenized units are joined with paragraph separators.
    unit_limit = min(max(1, body_max - 8), max(64, body_target // 2))
    for index, paragraph in enumerate(paragraphs):
        units.extend(split_paragraph(paragraph, index, unit_limit, tokenizer))
    if not units:
        return []

    chunks = []
    start = 0
    while start < len(units):
        end = start
        body_tokens = 0
        while end < len(units):
            candidate = units[end].tokens
            candidate_units = units[start:end + 1]
            candidate_body = "\n\n".join(unit.text for unit in candidate_units)
            exact_tokens = token_count(tokenizer, prefix + candidate_body)
            if end > start and exact_tokens > max_tokens:
                break
            body_tokens += candidate
            end += 1
            if exact_tokens >= target_tokens:
                break
        selected = units[start:end]
        body = "\n\n".join(unit.text for unit in selected)
        embedding_text = prefix + body
        chunks.append(Chunk(
            index=len(chunks), body_text=body, embedding_text=embedding_text,
            token_count=token_count(tokenizer, embedding_text),
            paragraph_start=selected[0].paragraph_index,
            paragraph_end=selected[-1].paragraph_index,
        ))
        if end >= len(units):
            break
        overlap_start = end
        overlap_size = 0
        while overlap_start > start + 1:
            candidate = units[overlap_start - 1].tokens
            if overlap_size + candidate > overlap_tokens:
                break
            overlap_start -= 1
            overlap_size += candidate
        start = overlap_start if overlap_start < end else end

    # Avoid weak tail embeddings. Merge a small final fragment into its
    # predecessor when the combined chunk still respects the hard maximum.
    while len(chunks) > 1 and chunks[-1].token_count < minimum_tokens:
        previous, tail = chunks[-2], chunks[-1]
        previous_parts = previous.body_text.split("\n\n")
        tail_parts = tail.body_text.split("\n\n")
        overlap_count = 0
        for size in range(1, min(len(previous_parts), len(tail_parts)) + 1):
            if previous_parts[-size:] == tail_parts[:size]:
                overlap_count = size
        combined_body = "\n\n".join(previous_parts + tail_parts[overlap_count:])
        combined_text = prefix + combined_body
        combined_tokens = token_count(tokenizer, combined_text)
        if combined_tokens > max_tokens:
            break
        chunks[-2] = Chunk(
            index=previous.index, body_text=combined_body, embedding_text=combined_text,
            token_count=combined_tokens, paragraph_start=previous.paragraph_start,
            paragraph_end=tail.paragraph_end,
        )
        chunks.pop()
    return chunks


def initialize_chunk_schema(db: sqlite3.Connection) -> None:
    db.executescript(
        """
        CREATE TABLE IF NOT EXISTS chunking_configs (
            version TEXT PRIMARY KEY,
            tokenizer TEXT NOT NULL,
            target_tokens INTEGER NOT NULL,
            max_tokens INTEGER NOT NULL,
            overlap_tokens INTEGER NOT NULL,
            config_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS article_chunks (
            id INTEGER PRIMARY KEY,
            article_id INTEGER NOT NULL REFERENCES articles(id) ON DELETE CASCADE,
            chunking_version TEXT NOT NULL REFERENCES chunking_configs(version),
            cleaning_version TEXT NOT NULL,
            source_hash TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            body_text TEXT NOT NULL,
            embedding_text TEXT NOT NULL,
            token_count INTEGER NOT NULL,
            paragraph_start INTEGER NOT NULL,
            paragraph_end INTEGER NOT NULL,
            chunk_hash TEXT NOT NULL UNIQUE,
            UNIQUE(article_id, chunking_version, chunk_index)
        );
        CREATE INDEX IF NOT EXISTS article_chunks_article_idx ON article_chunks(article_id);
        CREATE INDEX IF NOT EXISTS article_chunks_version_idx ON article_chunks(chunking_version);
        """
    )


def source_hash(title: str | None, text: str, cleaning_version: str) -> str:
    value = f"{cleaning_version}\0{title or ''}\0{text}"
    return hashlib.sha256(value.encode()).hexdigest()


def chunk_hash(version: str, article_id: int, index: int, text: str) -> str:
    value = f"{version}\0{article_id}\0{index}\0{text}"
    return hashlib.sha256(value.encode()).hexdigest()


def sync_chunks(
    db: sqlite3.Connection,
    tokenizer: Tokenizer,
    tokenizer_name: str,
    target: int,
    maximum: int,
    overlap: int,
    limit: int | None = None,
) -> tuple[int, int, int]:
    initialize_chunk_schema(db)
    config = {"tokenizer": tokenizer_name, "target_tokens": target,
              "max_tokens": maximum, "overlap_tokens": overlap}
    version = f"{CHUNKING_VERSION}-{target}-{maximum}-{overlap}"
    db.execute(
        "INSERT OR REPLACE INTO chunking_configs VALUES (?,?,?,?,?,?)",
        (version, tokenizer_name, target, maximum, overlap, json.dumps(config, sort_keys=True)),
    )
    sql = """SELECT id,title,article_text_clean,cleaning_version FROM articles
             WHERE quality_status='usable' AND article_text_clean IS NOT NULL ORDER BY id"""
    params: tuple = ()
    if limit is not None:
        sql += " LIMIT ?"
        params = (limit,)
    rows = db.execute(sql, params).fetchall()
    changed = skipped = total_chunks = 0
    started = time.monotonic()
    print(
        f"Preparing {len(rows):,} usable articles with {tokenizer_name} "
        f"(target={target}, max={maximum}, overlap={overlap})...",
        flush=True,
    )
    for processed, (article_id, title, text, cleaning_version) in enumerate(rows, 1):
        digest = source_hash(title, text, cleaning_version)
        existing = db.execute(
            "SELECT source_hash,COUNT(*) FROM article_chunks WHERE article_id=? AND chunking_version=?",
            (article_id, version),
        ).fetchone()
        if existing and existing[1] and existing[0] == digest:
            skipped += 1
            total_chunks += existing[1]
        else:
            chunks = make_chunks(title, text, tokenizer, target, maximum, overlap)
            db.execute("DELETE FROM article_chunks WHERE article_id=? AND chunking_version=?", (article_id, version))
            for chunk in chunks:
                db.execute(
                    """INSERT INTO article_chunks(article_id,chunking_version,cleaning_version,
                       source_hash,chunk_index,body_text,embedding_text,token_count,paragraph_start,
                       paragraph_end,chunk_hash) VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                    (article_id, version, cleaning_version, digest, chunk.index, chunk.body_text,
                     chunk.embedding_text, chunk.token_count, chunk.paragraph_start,
                     chunk.paragraph_end, chunk_hash(version, article_id, chunk.index, chunk.embedding_text)),
                )
            changed += 1
            total_chunks += len(chunks)
        if processed % 500 == 0 or processed == len(rows):
            db.commit()
            elapsed = max(time.monotonic() - started, 0.001)
            rate = processed / elapsed
            eta = (len(rows) - processed) / rate if rate else 0
            print(
                f"Processed {processed:,}/{len(rows):,} articles | "
                f"changed {changed:,} | skipped {skipped:,} | chunks {total_chunks:,} | "
                f"{rate:.1f} articles/s | ETA {eta / 60:.1f} min",
                flush=True,
            )
    # Remove stale chunks when an article no longer passes cleaning quality.
    cursor = db.execute(
        """DELETE FROM article_chunks WHERE chunking_version=? AND article_id NOT IN
           (SELECT id FROM articles WHERE quality_status='usable' AND article_text_clean IS NOT NULL)""",
        (version,),
    )
    if cursor.rowcount:
        print(f"Removed {cursor.rowcount:,} stale chunks", flush=True)
    db.commit()
    return changed, skipped, total_chunks


def load_tokenizer(name: str):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--target-tokens", type=int, default=500)
    parser.add_argument("--max-tokens", type=int, default=600)
    parser.add_argument("--overlap-tokens", type=int, default=75)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    tokenizer = load_tokenizer(args.tokenizer)
    with sqlite3.connect(args.db, timeout=30) as db:
        db.execute("PRAGMA foreign_keys=ON")
        changed, skipped, chunks = sync_chunks(
            db, tokenizer, args.tokenizer, args.target_tokens,
            args.max_tokens, args.overlap_tokens, args.limit,
        )
    print(f"Chunked {changed:,} articles; skipped {skipped:,} unchanged; {chunks:,} chunks total")


if __name__ == "__main__":
    main()
